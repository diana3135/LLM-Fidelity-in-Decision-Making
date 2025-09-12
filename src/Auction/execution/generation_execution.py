import re
import os
import pandas as pd
from typing import List, Dict, Optional
import anthropic
import openai
import concurrent.futures
import threading
import random
from tqdm import tqdm

# Data loading
human_experiment_data = pd.read_csv('../../human_experiment/auction_human_data.csv')
agent_experiment_data = human_experiment_data.copy()
umich_profiles = pd.read_excel("../profile_generation/umich_undergraduate_profiles.xlsx")
instructions_file_path = "experiment_instructions.txt"

def load_experiment_instructions(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

experiment_instructions = load_experiment_instructions(instructions_file_path)

# Helper Functions
def extract_bid_prices(bid_string: str) -> List[int]:
    return [int(x) for x in re.findall(r'd:(\d+)', bid_string)]

def calculate_profit(reserve_price: float, bid_prices: List[int]) -> float:
    if not bid_prices or bid_prices[0] < reserve_price:
        return 0
    if len(bid_prices) > 1 and bid_prices[1] >= reserve_price:
        return bid_prices[1]
    else:
        return reserve_price

def run_experiment_thread(simulator, profile_id, bidder_group_name, **kwargs):
    try:
        simulator.run_single_experiment(profile_id, bidder_group_name, **kwargs)
    except Exception as e:
        print(f"Error in {bidder_group_name} ({profile_id}): {e}")

# LLM Wrapper Classes
class ClaudeChatWrapper:
    def __init__(self, model="claude-3-5-sonnet-latest", anthropic_api_key=None, temperature=1.0):
        self.api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Please set ANTHROPIC_API_KEY in your environment or pass it in.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.temperature = temperature

    def invoke(self, system_prompt: str, user_prompt: str, temperature: float = 1.0):
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )

            if isinstance(response.content, list):
                content_str = "".join(item.text if hasattr(item, "text") else str(item) for item in response.content)
            elif isinstance(response.content, str):
                content_str = response.content.strip()
            else:
                raise ValueError("Unexpected response format from Anthropic API.")

            class ResponseObj:
                def __init__(self, content):
                    self.content = content

            return ResponseObj(content_str)

        except Exception as e:
            print(f"Error during Claude API call: {e}")
            return None

class OpenAIChatWrapper:
    def __init__(self, model="gpt-4o", openai_api_key=None):
        openai.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

    def invoke(self, system_prompt: str, user_prompt: str, temperature: float = 1.0):
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            
            content = response.choices[0].message.content
            
            class ResponseObj:
                def __init__(self, content):
                    self.content = content

            return ResponseObj(content)
        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            return None

class AgentState:
    def __init__(self, profile: Dict, instructions: str):
        self.profile_id: str = profile["id"]
        self.history: List[Dict] = []
        self.current_round: int = 1
        self.experiment_instructions: str = instructions
        self.age: str = profile["age"]
        self.gender: str = profile["gender"]
        self.race: str = profile["race"]
        self.program: str = profile["program"]

    def update(self, reserve_price: float, profit: float, num_bidders: int, bid_prices: List[int]):
        self.history.append({
            "reserve_price": reserve_price,
            "profit": profit,
            "round": self.current_round,
            "num_bidders": num_bidders,
            "bid_prices": bid_prices
        })
        self.current_round += 1

class AuctionExperiment:
    def __init__(self, participant_profile: Dict, bidder_group: pd.DataFrame, 
                 experiment_instructions: str, simulator, llm_type: str = "claude", 
                 model: str = None, risk_preference: str = None):
        
        self.state = AgentState(participant_profile, experiment_instructions)
        self.bidder_group = bidder_group
        self.simulator = simulator
        self.llm_type = llm_type.lower()
        
        # Initialize LLM
        if self.llm_type == "claude":
            default_model = "claude-3-5-sonnet-latest"
            self.llm = ClaudeChatWrapper(model=model or default_model, temperature=1.0)
        elif self.llm_type == "gpt":
            default_model = "gpt-4o"
            self.llm = OpenAIChatWrapper(model=model or default_model)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
        
        # Build system prompt
        base_identity = f"You are an undergraduate student at the University of Michigan.\nYou are {self.state.age}, {self.state.gender}, {self.state.race}, and studying {self.state.program}."
        
        risk_text = ""
        if risk_preference == "averse":
            risk_text = "You are a risk-averse decision maker, prioritizing lower-risk reservation prices to ensure positive profits."
        elif risk_preference == "seeking":
            risk_text = "You are a risk-seeking decision maker, prioritizing higher-risk reservation prices for the potential of higher profit."
        
        self.system_prompt = f"""
            {base_identity}
            {risk_text}

            You are about to participate in an experiment in the economics of decision making.

            Here are the experiment instructions:
            {self.state.experiment_instructions}

            IMPORTANT:
            - Try to **maximize your total profit** over 60 rounds.
            - You can only respond with an integer between 0 and 100 representing the reservation price.
            - Do not provide any explanation or additional text in your response.
        """

        self.user_prompt_template = """
            Here is your last round result:
            {last_round_info}

            Here is the history of all previous rounds:
            {history}
            
            Now it's round {current_round}.
            Number of Bidders in this round: {current_num_bidders}

            What reserve price do you set for *this round*?
        """


    def get_last_round_results_table(self) -> str:
        if not self.state.history:
            return "No previous round result.\n"

        last_round = self.state.history[-1]
        round_num = last_round["round"]
        reserve_price = last_round["reserve_price"]
        profit = last_round["profit"]
        num_bidders = last_round["num_bidders"]
        winning_price = profit

        bid_prices = last_round["bid_prices"]
        filtered_bids = [b for b in bid_prices if b >= reserve_price]
        filtered_bids.sort(reverse=True)

        if filtered_bids:
            table_str = "; ".join([
                f"Bidder: {i+1}, Drop-Out Price: {price}" 
                for i, price in enumerate(filtered_bids)
            ])
        else:
            table_str = "No bids were above your reserve price."

        result_str = (
            f"Round {round_num}:\n"
            f"  Your Reserve Price: {reserve_price}\n"
            f"  Profit: {profit}\n"
            f"  Number of Bidders: {num_bidders}\n"
            f"  Winning Price: {winning_price}\n\n"
            f"  Bids higher than your reserve price:\n"
            f"{table_str}\n"
        )
        return result_str

    def get_llm_decision(self, num_bidders: int, max_retries=3) -> Optional[int]:
        if not self.state.history:
            history_str = "No prior history."
        else:
            history_str = "\n".join([
                (
                    f"Round {h['round']}: "
                    f"Your Reserve Price={h['reserve_price']}, "
                    f"Your Profit={h['profit']}, "
                    f"Number of Bidders={h['num_bidders']}, "
                    f"Drop-Out Prices="
                    f"{[price if price != 0 else None for price in h['bid_prices']]} "
                )
                for h in self.state.history
            ])

        last_round_info = self.get_last_round_results_table()

        user_prompt = self.user_prompt_template.format(
            current_round=self.state.current_round,
            current_num_bidders=num_bidders, 
            last_round_info=last_round_info,
            history=history_str
        )


        for attempt in range(max_retries):
            try:
                llm_output = self.llm.invoke(
                    system_prompt=self.system_prompt,
                    user_prompt=user_prompt,
                    temperature=1.0
                ).content.strip()
                
                
                if re.fullmatch(r'\d+', llm_output):
                    return int(llm_output)
                else:
                    print(f"Invalid response on attempt {attempt + 1}: {llm_output}")
            except Exception as e:
                print(f"API call failed on attempt {attempt + 1}: {e}")
        
        return None

    def run_round(self):
        if self.state.current_round > len(self.bidder_group):
            return

        row = self.bidder_group.iloc[self.state.current_round - 1]
        bid_prices = extract_bid_prices(row['newBids'])
        num_bidders = int(row["numBidders"])
        reserve_price = self.get_llm_decision(num_bidders=num_bidders)

        if reserve_price is None:
            print(f"Skipping round {self.state.current_round} due to repeated failures.")
            return

        profit = calculate_profit(reserve_price, bid_prices)
        self.simulator.record_results(
            profile_id=self.state.profile_id,
            bidder_group_name=row["Bidder Group"],
            reserve_price=reserve_price,
            profit=profit
        )
        
        self.state.update(reserve_price, profit, num_bidders, bid_prices)

    def run_experiment(self):
        for _ in range(60):
            self.run_round()

class AuctionSimulator:
    def __init__(self, profiles: pd.DataFrame, experiment_data: pd.DataFrame, experiment_instructions: str):
        self.profiles = profiles
        self.experiment_data = experiment_data
        self.experiment_instructions = experiment_instructions
        self.results_df = pd.DataFrame(columns=["Bidder Group", "Profile ID", "profit_llm", "reserve_price_llm"])
        self.lock = threading.Lock()

    def generate_results_dataframe(self):
        with self.lock:
            return self.results_df.copy()

    def record_results(self, profile_id: str, bidder_group_name: str, reserve_price: float, profit: float):
        new_entry = pd.DataFrame({
            "Bidder Group": [bidder_group_name],
            "Profile ID": [profile_id],
            "reserve_price_llm": [reserve_price],
            "profit_llm": [profit]
        })
        
        with self.lock:
            self.results_df = pd.concat([self.results_df, new_entry], ignore_index=True)

    def run_single_experiment(self, profile_id: str, bidder_group_name: str, 
                            llm_type: str = "claude", model: str = None, 
                            risk_preference: str = None):
        profile_row = self.profiles[self.profiles["id"] == profile_id]
        if profile_row.empty:
            print(f"Error: Profile ID {profile_id} not found.")
            return
        
        profile = profile_row.iloc[0].to_dict()
        bidder_group = self.experiment_data[self.experiment_data["Bidder Group"] == bidder_group_name]
        if bidder_group.empty:
            print(f"Error: Bidder Group {bidder_group_name} not found.")
            return

        experiment = AuctionExperiment(
            profile, bidder_group, self.experiment_instructions, self,
            llm_type=llm_type, model=model, risk_preference=risk_preference
        )
        experiment.run_experiment()


def run_experiments(llm_type: str = "claude", model: str = None, risk_preference: str = None,
                   start_index: int = 1, end_index: int = 41,
                   output_file: str = None, max_workers: int = 1, use_threading: bool = False):
    
    simulator = AuctionSimulator(
        profiles=umich_profiles,
        experiment_data=agent_experiment_data,
        experiment_instructions=experiment_instructions
    )
    
    if output_file is None:
        output_file = f"{llm_type}_results.csv"

    experiment_params = {
        'llm_type': llm_type,
        'model': model,
        'risk_preference': risk_preference
    }

    try:
        if use_threading and max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i in range(start_index, end_index):
                    profile_id = f"B{i}"
                    bidder_group_name = f"S.{i}"
                    futures.append(executor.submit(
                        run_experiment_thread, simulator, profile_id, bidder_group_name, **experiment_params
                    ))
                
                concurrent.futures.wait(futures)
        else:
            for i in range(start_index, end_index):
                profile_id = f"B{i}"
                bidder_group_name = f"S.{i}"
                
                simulator.run_single_experiment(profile_id, bidder_group_name, **experiment_params)
                
                results_df = simulator.generate_results_dataframe()
                results_df.to_csv(output_file, index=False)

    
    except KeyboardInterrupt:
        print("\nInterrupted by user! Saving partial results...")
        results_df = simulator.generate_results_dataframe()
        results_df.to_csv(output_file, index=False)

# Imitation Experiment Classes and Functions

def extract_reserve_prices_from_response(response_text: str, context_num: int) -> List[int]:
    """Extract reserve prices from LLM response for imitation experiment."""
    prices = []
    for round_num in range(context_num + 1, 61):
        pattern = rf"round {round_num}:\s*(\d+)"
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            prices.append(int(match.group(1)))
        else:
            # If pattern not found, try to extract any number on the line
            line_pattern = rf".*{round_num}.*?(\d+)"
            line_match = re.search(line_pattern, response_text)
            if line_match:
                prices.append(int(line_match.group(1)))
            else:
                prices.append(None)  # Mark as failed to parse
    return prices

def prepare_auction_results_text(bid_info_group: pd.DataFrame) -> str:
    """Prepare human auction results text for context."""
    results = []
    for i, row in bid_info_group.iterrows():
        bid_prices = extract_bid_prices(row['newBids'])
        results.append(
            f"Round {i + 1}: "
            f"Your Reserve Price={row['rPrice']}, "
            f"Your Profit={row['myProfit']}, "
            f"Number of Bidders={row['numBidders']}, "
            f"Drop-Out Prices={bid_prices}"
        )
    return '\n'.join(results)

def apply_mode_transformation(text: str, mode: str) -> str:
    """Apply transformation based on imitation mode."""
    if mode == "direct":
        return text
    elif mode == "shuffle":
        lines = text.split('\n')
        random.shuffle(lines)
        return '\n'.join(lines)
    elif mode == "reverse":
        lines = text.split('\n')
        return '\n'.join(reversed(lines))
    elif mode == "mask":
        # Mask drop-out prices
        return re.sub(r'Drop-Out Prices=\[[^\]]*\]', 'Drop-Out Prices=[MASKED]', text)
    elif mode == "regionshuffle":
        # Shuffle within regions (first 15 and last 15 rounds separately)
        lines = text.split('\n')
        if len(lines) >= 30:
            first_half = lines[:15]
            second_half = lines[15:30]
            random.shuffle(first_half)
            random.shuffle(second_half)
            return '\n'.join(first_half + second_half)
    return text

class ImitationExperiment:
    def __init__(self, human_data: pd.DataFrame, experiment_instructions: str, 
                 context_num: int = 30, mode: str = "direct", llm_type: str = "gpt", 
                 model: str = None):
        self.human_data = human_data
        self.experiment_instructions = experiment_instructions
        self.context_num = context_num
        self.mode = mode
        self.llm_type = llm_type.lower()
        
        # Initialize LLM
        if self.llm_type == "claude":
            default_model = "claude-3-5-sonnet-latest"
            self.llm = ClaudeChatWrapper(model=model or default_model)
        elif self.llm_type == "gpt":
            default_model = "gpt-4o"
            self.llm = OpenAIChatWrapper(model=model or default_model)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    def run_imitation_for_bidder_group(self, bidder_group: str) -> pd.DataFrame:
        """Run imitation experiment for a single bidder group."""
        # Get human data for this bidder group
        group_data = self.human_data[self.human_data['Bidder Group'] == bidder_group].copy()
        if len(group_data) < 60:
            print(f"Skipping {bidder_group}: insufficient data ({len(group_data)} rounds)")
            return pd.DataFrame()
        
        # Prepare context data (first context_num rounds)
        context_data = group_data.iloc[:self.context_num].copy()
        
        # Prepare auction results text
        auction_results_text = prepare_auction_results_text(context_data)
        auction_results_text = apply_mode_transformation(auction_results_text, self.mode)
        
        # Create task description
        task = f"""
1. Review the participant's bidding results from the first {self.context_num} rounds.
2. For rounds {self.context_num + 1} to 60, continue their strategy by predicting reserve prices that match their decision patterns.
"""
        
        # Create prompts
        system_prompt = f"""
You are an AI agent participating in a simulated auction experiment.

{self.experiment_instructions}

## Task:
{task}

## Output Format:
Provide your responses in the following format, without any additional text or explanations:

round {self.context_num + 1}: [reserve_price]
round {self.context_num + 2}: [reserve_price]
...
round 60: [reserve_price]
"""
        
        user_prompt = f"""
## Participant's Auction Results (Rounds 1-{self.context_num}):
{auction_results_text}

## Upcoming Rounds Information:
{self._prepare_upcoming_rounds_info(group_data)}

Please predict the reserve prices for rounds {self.context_num + 1} to 60.
"""
        
        # Get LLM response
        try:
            response = self.llm.invoke(system_prompt, user_prompt, temperature=1.0)
            ai_reserve_prices = extract_reserve_prices_from_response(response.content, self.context_num)
            
            # Prepare results dataframe
            results = []
            for i, ai_price in enumerate(ai_reserve_prices):
                round_num = self.context_num + 1 + i
                if round_num <= 60 and ai_price is not None:
                    human_row = group_data.iloc[round_num - 1]
                    results.append({
                        'bidder_group': bidder_group,
                        'round': round_num,
                        'ai_reserve_price': ai_price,
                        'human_reserve_price': human_row['rPrice'],
                        'num_bidder': human_row['numBidders'],
                        'bid_prices': human_row['newBids'],
                        'mode': self.mode,
                        'context_num': self.context_num
                    })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            print(f"Error processing {bidder_group}: {e}")
            return pd.DataFrame()
    
    def _prepare_upcoming_rounds_info(self, group_data: pd.DataFrame) -> str:
        """Prepare information about upcoming rounds (bidder counts and bid prices)."""
        upcoming_info = []
        for i in range(self.context_num, min(60, len(group_data))):
            row = group_data.iloc[i]
            upcoming_info.append(f"Round {i + 1}: {row['numBidders']} bidders, bids: {row['newBids']}")
        return '\n'.join(upcoming_info)

def run_imitation_experiments(mode: str = "direct", context_num: int = 30, 
                            llm_type: str = "gpt", model: str = None,
                            bidder_groups: List[str] = None, 
                            output_file: str = None) -> pd.DataFrame:
    """Run imitation experiments for specified bidder groups."""
    
    if output_file is None:
        output_file = f"../../results/Auction/Imitation/{mode}.csv"
    
    # Initialize experiment
    experiment = ImitationExperiment(
        human_data=human_experiment_data,
        experiment_instructions=experiment_instructions,
        context_num=context_num,
        mode=mode,
        llm_type=llm_type,
        model=model
    )
    
    # Get bidder groups to process
    if bidder_groups is None:
        bidder_groups = human_experiment_data['Bidder Group'].unique()
    
    all_results = []
    processed_groups = []
    
    for bidder_group in tqdm(bidder_groups, desc=f"Processing {mode} mode"):
        try:
            result_df = experiment.run_imitation_for_bidder_group(bidder_group)
            if not result_df.empty:
                all_results.append(result_df)
                processed_groups.append(bidder_group)
                
                # Save intermediate results
                combined_results = pd.concat(all_results, ignore_index=True)
                combined_results.to_csv(output_file, index=False)
                
        except Exception as e:
            print(f"Error with bidder group {bidder_group}: {e}")
            continue
    
    # Final save
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        final_results.to_csv(output_file, index=False)
        return final_results
    else:
        return pd.DataFrame()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run auction experiments with different LLM configurations')
    parser.add_argument('--experiment', choices=['instruction', 'imitation'], default='instruction', 
                       help='Type of experiment to run')
    parser.add_argument('--llm', choices=['claude', 'gpt'], default='claude', help='LLM type to use')
    parser.add_argument('--model', type=str, help='Specific model to use (e.g., claude-3-5-sonnet-latest, gpt-4o)')
    parser.add_argument('--risk', choices=['averse', 'seeking'], help='Risk preference (instruction only)')
    parser.add_argument('--start', type=int, default=1, help='Start index for experiments (instruction only)')
    parser.add_argument('--end', type=int, default=41, help='End index for experiments (instruction only)')
    parser.add_argument('--output', type=str, help='Output file name')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers for threading (instruction only)')
    parser.add_argument('--threading', action='store_true', help='Use threading for parallel execution (instruction only)')
    
    # Imitation-specific arguments
    parser.add_argument('--mode', choices=['direct', 'shuffle', 'reverse', 'mask', 'regionshuffle'], 
                       default='direct', help='Imitation mode (imitation only)')
    parser.add_argument('--context-num', type=int, default=30, 
                       help='Number of context rounds for imitation (imitation only)')
    parser.add_argument('--bidder-groups', nargs='*', 
                       help='Specific bidder groups to process (imitation only)')
    
    args = parser.parse_args()
    
    if args.experiment == 'instruction':
        run_experiments(
            llm_type=args.llm,
            model=args.model,
            risk_preference=args.risk,
            start_index=args.start,
            end_index=args.end,
            output_file=args.output,
            max_workers=args.workers,
            use_threading=args.threading
        )
    elif args.experiment == 'imitation':
        run_imitation_experiments(
            mode=args.mode,
            context_num=args.context_num,
            llm_type=args.llm,
            model=args.model,
            bidder_groups=args.bidder_groups,
            output_file=args.output
        )

if __name__ == "__main__":
    main()
