import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import itertools
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class EggCollectionSimulator:
    """
    Simulates an automated egg collection system for a 10,080-hen farm
    with 18 conveyor belts, optimizing for breakage and energy consumption.
    """
    
    def __init__(self):
        # System constants
        self.num_belts = 18
        self.belt_length = 12  # meters per belt
        self.motor_power = 0.56  # kW (0.75 HP = 0.56 kW)
        self.egg_cost = 25  # Rs per egg
        self.energy_cost = 25  # Rs per kWh (current domestic tariff)
        self.daily_hens = 10080
        
        # Temperature effects on production
        self.temp_production = {
            25: 0.88,  # 88% production at 25°C
            32: 0.80,  # 80% production at 32°C
            35: 0.73   # 73% production at 35°C
        }
        
        # Base breakage rates by temperature
        self.base_breakage = {
            25: 0.025,  # 2.5% at 25°C
            32: 0.035,  # 3.5% at 32°C
            35: 0.048   # 4.8% at 35°C
        }
        
        # Time periods in minutes
        self.day_minutes = 1440
        self.time_slots = {
            'morning_peak': (360, 600),     # 6:00-10:00 (4 hours)
            'midday': (660, 900),           # 11:00-15:00 (4 hours)
            'afternoon': (960, 1140)        # 16:00-19:00 (3 hours)
        }
        
        # Egg arrival rate patterns (eggs per minute per belt)
        # Derived from laying patterns: 60% in morning, 25% midday, 15% afternoon
        self.arrival_pattern = {
            'morning_peak': 0.60,
            'midday': 0.25,
            'afternoon': 0.15
        }
        
        # Speed settings (m/s)
        self.speed_levels = {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7
        }
        
        # Collection frequency schedules
        self.collection_schedules = {
            3: [540, 780, 1020],    # 9:00, 13:00, 17:00
            4: [480, 660, 840, 1020],  # 8:00, 11:00, 14:00, 17:00
            5: [480, 600, 720, 840, 960],  # 8:00, 10:00, 12:00, 14:00, 16:00
            6: [480, 600, 720, 840, 960, 1080]  # 8:00, 10:00, 12:00, 14:00, 16:00, 18:00
        }
        
        # Energy consumption factors
        self.energy_per_speed = {
            'low': 0.7,    # 70% of max power
            'medium': 0.85, # 85% of max power
            'high': 1.0     # 100% of max power
        }
        
        # Vibration breakage factor (increases with speed)
        self.vibration_factor = {
            'low': 0.005,
            'medium': 0.015,
            'high': 0.035
        }
    
    def calculate_arrival_rates(self, temperature: int, speed_level: str) -> Dict:
        """
        Calculate egg arrival rates based on temperature and time of day.
        Uses Poisson process with time-varying rates.
        """
        # Total daily eggs for one belt
        hens_per_belt = self.daily_hens / self.num_belts
        daily_eggs_per_belt = hens_per_belt * self.temp_production[temperature]
        
        rates = {}
        for period, proportion in self.arrival_pattern.items():
            start, end = self.time_slots[period]
            duration = (end - start) / 60  # hours
            eggs_in_period = daily_eggs_per_belt * proportion
            
            # Eggs per minute in this period
            rate_per_minute = eggs_in_period / (duration * 60)
            
            # Adjust rate slightly based on temperature (hens lay differently in heat)
            if temperature > 30:
                rate_per_minute *= 0.95  # Slightly slower laying in heat
            
            rates[period] = rate_per_minute
        
        return rates
    
    def simulate_breakage(self, queue_length: float, temperature: int, 
                         speed_level: str, belt_utilization: float) -> float:
        """
        Calculate breakage probability based on multiple factors.
        """
        # Base breakage
        breakage_prob = self.base_breakage[temperature]
        
        # Accumulation breakage (increases with queue length)
        # Assuming belt capacity is 200 eggs (based on 12m length, 6cm per egg spacing)
        max_capacity = 200
        accumulation_factor = min(queue_length / max_capacity, 1.0) * 0.04
        
        # Vibration breakage (increases with speed)
        vibration_breakage = self.vibration_factor[speed_level]
        
        # Belt utilization effect (higher utilization = more collisions)
        utilization_factor = belt_utilization * 0.02
        
        # Temperature acceleration factor
        temp_factor = max(0, (temperature - 25) / 10) * 0.01
        
        total_breakage = (breakage_prob + accumulation_factor + 
                         vibration_breakage + utilization_factor + temp_factor)
        
        # Cap at reasonable maximum
        return min(total_breakage, 0.15)
    
    def simulate_belt_operation(self, temperature: int, speed_level: str, 
                               frequency: int, dynamic_speed: bool = False) -> Dict:
        """
        Simulate one day of operation for one belt.
        """
        # Get arrival rates
        arrival_rates = self.calculate_arrival_rates(temperature, speed_level)
        
        # Initialize variables
        time = 0
        queue = 0  # eggs on belt
        total_eggs_arrived = 0
        total_eggs_broken = 0
        total_eggs_collected = 0
        belt_running = False
        running_time = 0
        collection_times = self.collection_schedules[frequency]
        
        # Track queue over time for statistics
        queue_history = []
        breakage_history = []
        
        # Main simulation loop (minute by minute)
        while time < self.day_minutes:
            # Determine current period
            current_period = None
            for period, (start, end) in self.time_slots.items():
                if start <= time < end:
                    current_period = period
                    break
            
            # If no specific period, use baseline rate
            if current_period is None:
                arrival_rate = arrival_rates['morning_peak'] * 0.1  # Very low rate
            else:
                arrival_rate = arrival_rates[current_period]
            
            # Eggs arriving this minute (Poisson process)
            eggs_arriving = np.random.poisson(arrival_rate)
            total_eggs_arrived += eggs_arriving
            
            # Add to queue
            queue += eggs_arriving
            
            # Dynamic speed adjustment
            current_speed = speed_level
            if dynamic_speed:
                if 'morning_peak' in str(current_period):
                    current_speed = 'high'
                elif 'midday' in str(current_period):
                    current_speed = 'medium'
                else:
                    current_speed = 'low'
            
            # Check if belt should be running (collection time)
            belt_running = any(abs(time - ct) < 30 for ct in collection_times)  # Run for 30 mins around collection
            
            if belt_running:
                running_time += 1
                
                # Belt clears eggs based on speed and egg spacing (0.06 m per egg)
                # speed_levels are in m/s; convert to meters per minute then to eggs per minute
                speed = self.speed_levels[current_speed]  # m/s
                meters_per_min = speed * 60  # m per minute
                eggs_per_min_capacity = meters_per_min / 0.06  # eggs per minute, assuming 6cm spacing
                eggs_cleared = min(queue, eggs_per_min_capacity)
                
                # Calculate breakage for eggs on belt
                belt_utilization = queue / 200  # Simplified capacity
                breakage_prob = self.simulate_breakage(queue, temperature, 
                                                      current_speed, belt_utilization)
                
                # Apply breakage to eggs being transported/cleared this minute
                eggs_broken = int(eggs_cleared * breakage_prob)
                eggs_broken = min(eggs_broken, int(queue))  # Can't break more than available
                
                total_eggs_broken += eggs_broken
                total_eggs_collected += eggs_cleared - eggs_broken
                
                # Update queue
                queue = max(0, queue - eggs_cleared)
                
                # Record history
                breakage_history.append(breakage_prob)
            
            queue_history.append(queue)
            time += 1
        
        # Calculate metrics
        avg_queue = np.mean(queue_history)
        max_queue = np.max(queue_history)
        avg_breakage = np.mean(breakage_history) if breakage_history else 0
        
        # Energy consumption
        energy_per_hour = self.motor_power * self.energy_per_speed[speed_level]
        energy_kwh = (running_time / 60) * energy_per_hour
        
        # Costs
        breakage_cost = total_eggs_broken * self.egg_cost
        energy_cost = energy_kwh * self.energy_cost
        total_cost = breakage_cost + energy_cost
        
        return {
            'temperature': temperature,
            'speed': speed_level,
            'frequency': frequency,
            'total_eggs': total_eggs_arrived,
            'broken_eggs': total_eggs_broken,
            'breakage_rate': total_eggs_broken / max(total_eggs_arrived, 1),
            'avg_queue': avg_queue,
            'max_queue': max_queue,
            'running_time_min': running_time,
            'energy_kwh': energy_kwh,
            'breakage_cost': breakage_cost,
            'energy_cost': energy_cost,
            'total_cost': total_cost,
            'avg_breakage_prob': avg_breakage
            ,
            'queue_history': queue_history,
            'breakage_history': breakage_history
        }
    
    def run_scenario(self, scenario_name: str, params: Dict, runs: int = 10) -> Dict:
        """
        Run a scenario multiple times for statistical significance.
        """
        results = []
        
        for _ in range(runs):
            result = self.simulate_belt_operation(
                temperature=params.get('temperature', 25),
                speed_level=params.get('speed', 'medium'),
                frequency=params.get('frequency', 3),
                dynamic_speed=params.get('dynamic_speed', False)
            )
            results.append(result)
        
        # Aggregate results
        df = pd.DataFrame(results)
        aggregated = {
            'scenario': scenario_name,
            'avg_breakage_rate': df['breakage_rate'].mean(),
            'std_breakage_rate': df['breakage_rate'].std(),
            'avg_energy_kwh': df['energy_kwh'].mean(),
            'std_energy_kwh': df['energy_kwh'].std(),
            'avg_total_cost': df['total_cost'].mean(),
            'std_total_cost': df['total_cost'].std(),
            'avg_queue': df['avg_queue'].mean(),
            'max_queue': df['max_queue'].mean(),
            'avg_running_time': df['running_time_min'].mean() / 60,  # hours
            'temperature': params.get('temperature', 25),
            'speed': params.get('speed', 'medium'),
            'frequency': params.get('frequency', 3)
        }
        
        return aggregated
    
    def run_all_scenarios(self) -> pd.DataFrame:
        """
        Run all defined scenarios.
        """
        scenarios = []
        
        # Baseline scenarios (B0)
        print("Running Baseline Scenarios...")
        scenarios.append(self.run_scenario('B0_25C_Fixed', {'temperature': 25, 'speed': 'medium', 'frequency': 3}))
        scenarios.append(self.run_scenario('B0_35C_Fixed', {'temperature': 35, 'speed': 'medium', 'frequency': 3}))
        
        # Temperature-based scenarios (T1-T3)
        print("Running Temperature Scenarios...")
        for temp, speed in [(25, 'low'), (25, 'medium'), (25, 'high')]:
            scenarios.append(self.run_scenario(f'T1_25C_{speed}', 
                                              {'temperature': 25, 'speed': speed, 'frequency': 3}))
        
        for temp, speed in [(32, 'low'), (32, 'medium'), (32, 'high')]:
            scenarios.append(self.run_scenario(f'T2_32C_{speed}', 
                                              {'temperature': 32, 'speed': speed, 'frequency': 4}))
        
        for temp, speed in [(35, 'low'), (35, 'medium'), (35, 'high')]:
            scenarios.append(self.run_scenario(f'T3_35C_{speed}', 
                                              {'temperature': 35, 'speed': speed, 'frequency': 5}))
        
        # Time-of-day scenarios (D1-D3)
        print("Running Time-of-Day Scenarios...")
        scenarios.append(self.run_scenario('D1_MorningPeak_Fixed', 
                                          {'temperature': 25, 'speed': 'medium', 'frequency': 3}))
        scenarios.append(self.run_scenario('D1_MorningPeak_Dynamic', 
                                          {'temperature': 25, 'speed': 'medium', 'frequency': 3, 'dynamic_speed': True}))
        
        scenarios.append(self.run_scenario('D2_Midday_Fixed', 
                                          {'temperature': 25, 'speed': 'medium', 'frequency': 3}))
        scenarios.append(self.run_scenario('D2_Midday_Dynamic', 
                                          {'temperature': 25, 'speed': 'low', 'frequency': 3}))
        
        scenarios.append(self.run_scenario('D3_FullDay_Fixed', 
                                          {'temperature': 25, 'speed': 'medium', 'frequency': 3}))
        scenarios.append(self.run_scenario('D3_FullDay_Dynamic', 
                                          {'temperature': 25, 'speed': 'medium', 'frequency': 3, 'dynamic_speed': True}))
        
        # Speed optimization scenarios (S1-S3) across temperatures
        print("Running Speed Optimization Scenarios...")
        for temp in [25, 32, 35]:
            for speed in ['low', 'medium', 'high']:
                freq = 3 if temp == 25 else (4 if temp == 32 else 5)
                scenarios.append(self.run_scenario(f'S_{temp}C_{speed}', 
                                                  {'temperature': temp, 'speed': speed, 'frequency': freq}))
        
        # Frequency optimization scenarios (F1-F3)
        print("Running Frequency Optimization Scenarios...")
        for freq in [3, 4, 6]:
            scenarios.append(self.run_scenario(f'F{freq}_25C', 
                                              {'temperature': 25, 'speed': 'medium', 'frequency': freq}))
            scenarios.append(self.run_scenario(f'F{freq}_35C', 
                                              {'temperature': 35, 'speed': 'medium', 'frequency': freq}))
        
        # Scale results to full farm (18 belts, 30 days)
        df = pd.DataFrame(scenarios)
        
        # Scale to full farm
        df['farm_breakage_cost'] = df['avg_breakage_rate'] * (self.daily_hens * 0.85) * 30 * self.egg_cost
        df['farm_energy_cost'] = df['avg_energy_kwh'] * self.num_belts * 30 * self.energy_cost
        df['farm_total_cost'] = df['farm_breakage_cost'] + df['farm_energy_cost']
        
        return df
    
    def plot_results(self, results_df: pd.DataFrame):
        """
        Create visualization of results.
        """
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # 1. Breakage vs Temperature
        temp_groups = results_df.groupby('temperature')['avg_breakage_rate'].mean()
        axes[0,0].bar(temp_groups.index, temp_groups.values)
        axes[0,0].set_xlabel('Temperature (°C)')
        axes[0,0].set_ylabel('Breakage Rate')
        axes[0,0].set_title('Breakage Rate vs Temperature')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Energy vs Speed
        speed_order = ['low', 'medium', 'high']
        speed_energy = []
        for speed in speed_order:
            speed_data = results_df[results_df['speed'] == speed]
            speed_energy.append(speed_data['avg_energy_kwh'].mean())
        
        axes[0,1].bar(speed_order, speed_energy)
        axes[0,1].set_xlabel('Belt Speed')
        axes[0,1].set_ylabel('Energy (kWh/belt/day)')
        axes[0,1].set_title('Energy Consumption vs Belt Speed')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Breakage vs Speed at different temperatures
        for temp in [25, 32, 35]:
            temp_data = results_df[results_df['temperature'] == temp]
            speed_breakage = []
            for speed in speed_order:
                speed_breakage.append(temp_data[temp_data['speed'] == speed]['avg_breakage_rate'].mean())
            axes[0,2].plot(speed_order, speed_breakage, marker='o', label=f'{temp}°C')
        
        axes[0,2].set_xlabel('Belt Speed')
        axes[0,2].set_ylabel('Breakage Rate')
        axes[0,2].set_title('Breakage vs Speed at Different Temperatures')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Total Cost vs Speed at 35°C
        hot_data = results_df[results_df['temperature'] == 35]
        speed_cost = []
        for speed in speed_order:
            cost = hot_data[hot_data['speed'] == speed]['farm_total_cost'].mean()
            speed_cost.append(cost)
        
        axes[1,0].bar(speed_order, speed_cost)
        axes[1,0].set_xlabel('Belt Speed')
        axes[1,0].set_ylabel('Total Monthly Cost (Rs)')
        axes[1,0].set_title('Total Cost vs Speed at 35°C')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Breakage vs Collection Frequency
        freq_groups = results_df.groupby('frequency')['avg_breakage_rate'].mean()
        axes[1,1].bar(freq_groups.index, freq_groups.values)
        axes[1,1].set_xlabel('Collection Frequency (runs/day)')
        axes[1,1].set_ylabel('Breakage Rate')
        axes[1,1].set_title('Breakage vs Collection Frequency')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Dynamic vs Fixed Speed Comparison
        dynamic_scenarios = ['D1_MorningPeak_Dynamic', 'D3_FullDay_Dynamic']
        fixed_scenarios = ['D1_MorningPeak_Fixed', 'D3_FullDay_Fixed']
        
        dynamic_breakage = []
        fixed_breakage = []
        for dyn, fix in zip(dynamic_scenarios, fixed_scenarios):
            dyn_data = results_df[results_df['scenario'] == dyn]
            fix_data = results_df[results_df['scenario'] == fix]
            if not dyn_data.empty:
                dynamic_breakage.append(dyn_data['avg_breakage_rate'].values[0])
            if not fix_data.empty:
                fixed_breakage.append(fix_data['avg_breakage_rate'].values[0])
        
        x = np.arange(len(dynamic_scenarios))
        width = 0.35
        axes[1,2].bar(x - width/2, fixed_breakage, width, label='Fixed Speed')
        axes[1,2].bar(x + width/2, dynamic_breakage, width, label='Dynamic Speed')
        axes[1,2].set_xlabel('Scenario')
        axes[1,2].set_ylabel('Breakage Rate')
        axes[1,2].set_title('Dynamic vs Fixed Speed Control')
        axes[1,2].set_xticks(x)
        axes[1,2].set_xticklabels(['Morning Peak', 'Full Day'])
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        # 7. Cost Breakdown for Optimal Hot Day Strategy
        hot_optimal = results_df[
            (results_df['temperature'] == 35) & 
            (results_df['speed'] == 'medium') & 
            (results_df['frequency'] == 5)
        ]
        
        if not hot_optimal.empty:
            breakage_cost = hot_optimal['farm_breakage_cost'].values[0]
            energy_cost = hot_optimal['farm_energy_cost'].values[0]
            
            axes[2,0].pie([breakage_cost, energy_cost], 
                         labels=['Breakage Cost', 'Energy Cost'],
                         autopct='%1.1f%%')
            axes[2,0].set_title('Cost Breakdown: Optimal Hot Day Strategy')
        
        # 8. Trade-off Curve: Breakage vs Energy
        scatter_data = results_df[['avg_breakage_rate', 'avg_energy_kwh']].dropna()
        axes[2,1].scatter(scatter_data['avg_energy_kwh'], 
                         scatter_data['avg_breakage_rate'], 
                         alpha=0.6)
        axes[2,1].set_xlabel('Energy Consumption (kWh/belt/day)')
        axes[2,1].set_ylabel('Breakage Rate')
        axes[2,1].set_title('Breakage vs Energy Trade-off')
        axes[2,1].grid(True, alpha=0.3)
        
        # 9. Queue Length Over Time (sample simulation)
        # Run a sample simulation to get queue history
        sample_result = self.simulate_belt_operation(35, 'medium', 5)
        time_points = np.arange(len(sample_result.get('queue_history', []))) / 60
        
        axes[2,2].plot(time_points[:100], sample_result.get('queue_history', [])[:100])
        axes[2,2].set_xlabel('Time (hours)')
        axes[2,2].set_ylabel('Queue Length (eggs on belt)')
        axes[2,2].set_title('Sample Queue Length Over Time (35°C)')
        axes[2,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('simulation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_optimal_strategy(self, results_df: pd.DataFrame) -> Dict:
        """
        Generate optimal strategy based on simulation results.
        """
        # Find minimum cost strategy for hot days
        hot_days = results_df[results_df['temperature'] == 35]
        if not hot_days.empty:
            optimal_row = hot_days.loc[hot_days['farm_total_cost'].idxmin()]
            
            optimal_strategy = {
                'temperature': 35,
                'optimal_speed': optimal_row['speed'],
                'optimal_frequency': int(optimal_row['frequency']),
                'expected_breakage_rate': optimal_row['avg_breakage_rate'],
                'expected_energy_kwh': optimal_row['avg_energy_kwh'],
                'monthly_breakage_cost': optimal_row['farm_breakage_cost'],
                'monthly_energy_cost': optimal_row['farm_energy_cost'],
                'monthly_total_cost': optimal_row['farm_total_cost'],
                'recommendations': [
                    f"Use {optimal_row['speed']} belt speed during hot days",
                    f"Collect eggs {int(optimal_row['frequency'])} times per day",
                    f"Implement dynamic speed control during morning peak",
                    f"Schedule runs at: {self.collection_schedules[int(optimal_row['frequency'])]} minutes past midnight"
                ]
            }
            
            # Calculate savings vs baseline
            baseline_hot = results_df[
                (results_df['scenario'] == 'B0_35C_Fixed') | 
                (results_df['temperature'] == 35) & 
                (results_df['speed'] == 'medium') & 
                (results_df['frequency'] == 3)
            ]
            
            if not baseline_hot.empty:
                baseline_cost = baseline_hot['farm_total_cost'].mean()
                savings = baseline_cost - optimal_row['farm_total_cost']
                savings_percent = (savings / baseline_cost) * 100
                
                optimal_strategy['savings_vs_baseline'] = savings
                optimal_strategy['savings_percent'] = savings_percent
                optimal_strategy['baseline_cost'] = baseline_cost
            
            return optimal_strategy
        
        return {}

# Main execution
def main():
    print("Starting Egg Collection System Simulation...")
    print("=" * 60)
    
    # Initialize simulator
    simulator = EggCollectionSimulator()
    
    # Run all scenarios
    results = simulator.run_all_scenarios()
    
    # Save results to CSV
    results.to_csv('simulation_results.csv', index=False)
    print("\nResults saved to 'simulation_results.csv'")
    
    # Display key results
    print("\n" + "=" * 60)
    print("KEY SIMULATION RESULTS")
    print("=" * 60)
    
    # Baseline comparison
    baseline_25 = results[results['scenario'] == 'B0_25C_Fixed']
    baseline_35 = results[results['scenario'] == 'B0_35C_Fixed']
    
    if not baseline_25.empty and not baseline_35.empty:
        print(f"\n1. Baseline at 25°C:")
        print(f"   Breakage Rate: {baseline_25['avg_breakage_rate'].values[0]:.3%}")
        print(f"   Energy Consumption: {baseline_25['avg_energy_kwh'].values[0]:.2f} kWh/belt/day")
        print(f"   Monthly Cost: Rs {baseline_25['farm_total_cost'].values[0]:,.0f}")
        
        print(f"\n2. Baseline at 35°C (Hot Day):")
        print(f"   Breakage Rate: {baseline_35['avg_breakage_rate'].values[0]:.3%}")
        print(f"   Energy Consumption: {baseline_35['avg_energy_kwh'].values[0]:.2f} kWh/belt/day")
        print(f"   Monthly Cost: Rs {baseline_35['farm_total_cost'].values[0]:,.0f}")
        
        cost_increase = ((baseline_35['farm_total_cost'].values[0] - 
                         baseline_25['farm_total_cost'].values[0]) / 
                        baseline_25['farm_total_cost'].values[0]) * 100
        print(f"   Cost Increase in Heat: {cost_increase:.1f}%")
    
    # Find optimal strategies
    print(f"\n3. Optimal Strategy Analysis:")
    
    # For 25°C
    temp_25 = results[results['temperature'] == 25]
    if not temp_25.empty:
        optimal_25 = temp_25.loc[temp_25['farm_total_cost'].idxmin()]
        print(f"   At 25°C: Speed={optimal_25['speed']}, Freq={optimal_25['frequency']}")
        print(f"   Breakage: {optimal_25['avg_breakage_rate']:.3%}, "
              f"Cost: Rs {optimal_25['farm_total_cost']:,.0f}")
    
    # For 35°C
    temp_35 = results[results['temperature'] == 35]
    if not temp_35.empty:
        optimal_35 = temp_35.loc[temp_35['farm_total_cost'].idxmin()]
        print(f"   At 35°C: Speed={optimal_35['speed']}, Freq={optimal_35['frequency']}")
        print(f"   Breakage: {optimal_35['avg_breakage_rate']:.3%}, "
              f"Cost: Rs {optimal_35['farm_total_cost']:,.0f}")
    
    # Dynamic vs fixed comparison
    dynamic_benefit = results[
        results['scenario'].str.contains('Dynamic')
    ]['avg_breakage_rate'].mean()
    
    fixed_breakage = results[
        results['scenario'].str.contains('Fixed') & 
        ~results['scenario'].str.contains('Dynamic')
    ]['avg_breakage_rate'].mean()
    
    if not np.isnan(dynamic_benefit) and not np.isnan(fixed_breakage):
        reduction = ((fixed_breakage - dynamic_benefit) / fixed_breakage) * 100
        print(f"\n4. Dynamic Speed Control Benefit:")
        print(f"   Breakage Reduction: {reduction:.1f}%")
    
    # Generate optimal strategy
    optimal_strategy = simulator.generate_optimal_strategy(results)
    if optimal_strategy:
        print(f"\n5. RECOMMENDED OPTIMAL STRATEGY FOR HOT DAYS:")
        print(f"   Temperature: {optimal_strategy['temperature']}°C")
        print(f"   Belt Speed: {optimal_strategy['optimal_speed']}")
        print(f"   Collection Frequency: {optimal_strategy['optimal_frequency']} times/day")
        print(f"   Expected Breakage: {optimal_strategy['expected_breakage_rate']:.3%}")
        print(f"   Expected Energy: {optimal_strategy['expected_energy_kwh']:.2f} kWh/belt/day")
        print(f"   Monthly Total Cost: Rs {optimal_strategy['monthly_total_cost']:,.0f}")
        
        if 'savings_percent' in optimal_strategy:
            print(f"   Savings vs Baseline: {optimal_strategy['savings_percent']:.1f}% "
                  f"(Rs {optimal_strategy['savings_vs_baseline']:,.0f})")
        
        print(f"\n   Recommendations:")
        for i, rec in enumerate(optimal_strategy['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    simulator.plot_results(results)
    
    return results, optimal_strategy, simulator

if __name__ == "__main__":
    # Run simulation
    simulation_results, optimal_strategy, simulator = main()
    
    # Create summary report
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY REPORT")
    print("=" * 60)
    
    # Save optimal strategy to file
    if optimal_strategy:
        with open('optimal_strategy.txt', 'w') as f:
            f.write("OPTIMAL EGG COLLECTION STRATEGY FOR HOT DAYS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Temperature: {optimal_strategy['temperature']}°C\n")
            f.write(f"Optimal Belt Speed: {optimal_strategy['optimal_speed']}\n")
            f.write(f"Collection Frequency: {optimal_strategy['optimal_frequency']} times/day\n")
            f.write(f"Schedule Times: {simulator.collection_schedules[optimal_strategy['optimal_frequency']]} minutes past midnight\n")
            f.write(f"Expected Breakage Rate: {optimal_strategy['expected_breakage_rate']:.3%}\n")
            f.write(f"Expected Energy Consumption: {optimal_strategy['expected_energy_kwh']:.2f} kWh/belt/day\n")
            f.write(f"Monthly Farm Cost: Rs {optimal_strategy['monthly_total_cost']:,.0f}\n\n")
            
            if 'savings_percent' in optimal_strategy:
                f.write(f"SAVINGS vs BASELINE:\n")
                f.write(f"  Reduction: {optimal_strategy['savings_percent']:.1f}%\n")
                f.write(f"  Amount: Rs {optimal_strategy['savings_vs_baseline']:,.0f}\n\n")
            
            f.write("RECOMMENDATIONS:\n")
            for rec in optimal_strategy['recommendations']:
                f.write(f"  • {rec}\n")
            
            f.write("\nADDITIONAL RECOMMENDATIONS:\n")
            f.write("  • Monitor temperature and adjust speed dynamically\n")
            f.write("  • Implement morning peak detection for automatic speed adjustment\n")
            f.write("  • Consider variable frequency drives for motors to save energy\n")
            f.write("  • Regular maintenance of belts to reduce vibration breakage\n")
        
        print("\nOptimal strategy saved to 'optimal_strategy.txt'")