import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as st
import pandas as pd

# Ensure output directory exists
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Simulation Code for Determining the Number of Oracle Nodes

# Magic numbers (constants)
INITIAL_TRUST_SCORE = 0.65  # Starting trust score for all oracle nodes
TRUST_SCORE_THRESHOLD = 0.55  # Minimum trust score required for a node to remain active
TRUST_SCORE_CAP = 0.95  # Maximum trust score to prevent overconfidence
DESIRED_PROBABILITY = 0.999  # Target reliability probability for the system

# Node type probabilities
MALICIOUS_PROBABILITY = 0.01  # Probability of a node being malicious
INCOMPETENT_PROBABILITY = 0.09  # Probability of a node being incompetent
COMPETENT_PROBABILITY = 1 - MALICIOUS_PROBABILITY - INCOMPETENT_PROBABILITY  # Remaining nodes are competent

# Accuracy scores by node type
MALICIOUS_MEAN = 0.25  # Mean accuracy for malicious nodes
INCOMPETENT_MEAN = 0.55  # Mean accuracy for incompetent nodes
COMPETENT_MEAN = 0.9  # Mean accuracy for competent nodes
STD_DEV = 0.05  # Standard deviation for accuracy scores

EXP_WEIGHT_DECAY = 0.9  # Exponential decay factor for trust score updates

NUM_ORACLE_NODES = 1000  # Number of oracle nodes to initialize

class OracleNode:
    """
    Represents an individual oracle node with attributes for type, accuracy, and trust score.
    """
    def __init__(self):
        self.trust_score = INITIAL_TRUST_SCORE  # Initial trust score
        self.type = self._assign_type()  # Node type: malicious, incompetent, or competent
        self.accuracy_score = self._assign_accuracy_score()  # Accuracy score based on type
        self.times_picked = 0  # Number of times the node has been selected for testing
        self.previous_test_results = []  # History of recent test results

    def _assign_type(self):
        """Randomly assigns a type to the node based on predefined probabilities."""
        rand = np.random.rand()
        if rand < MALICIOUS_PROBABILITY:
            return 'malicious'
        elif rand < MALICIOUS_PROBABILITY + INCOMPETENT_PROBABILITY:
            return 'incompetent'
        else:
            return 'competent'

    def _assign_accuracy_score(self):
        """Assigns an accuracy score to the node based on its type."""
        if self.type == 'malicious':
            return np.random.normal(MALICIOUS_MEAN, STD_DEV)
        elif self.type == 'incompetent':
            return np.random.normal(INCOMPETENT_MEAN, STD_DEV)
        else:
            return np.random.normal(COMPETENT_MEAN, STD_DEV)

    def pick(self, correct):
        """Updates the node's test history and recalculates its trust score."""
        self.times_picked += 1
        self.previous_test_results.append(correct)  # Record the result of the test
        if len(self.previous_test_results) > 10:  # Limit history to the last 10 tests
            self.previous_test_results.pop(0)
        if self.times_picked % 10 == 0:  # Update trust score every 10 picks
            self.update_trust_score()

    def update_trust_score(self):
        """Calculates the trust score using an exponential weighted average."""
        weighted_sum = 0
        weight = 1
        for result in reversed(self.previous_test_results):
            weighted_sum += result * weight
            weight *= EXP_WEIGHT_DECAY
        self.trust_score = weighted_sum / sum(EXP_WEIGHT_DECAY ** i for i in range(len(self.previous_test_results)))
        self.trust_score = min(max(self.trust_score, 0), TRUST_SCORE_CAP)  # Clamp trust score within bounds

    def is_active(self):
        """Determines if the node is active based on its trust score."""
        return self.trust_score >= TRUST_SCORE_THRESHOLD

    def weight(self):
        """Returns the selection weight of the node, proportional to the square of its trust score."""
        return self.trust_score ** 2

    def __repr__(self):
        return f"OracleNode(trust_score={self.trust_score:.2f}, accuracy_score={self.accuracy_score:.2f}, type={self.type}, times_picked={self.times_picked})"

# Initialize a global list of oracle nodes
oracle_nodes = [OracleNode() for _ in range(NUM_ORACLE_NODES)]

class OracleTest:
    """
    Manages individual tests to assess the performance and reliability of oracle nodes.
    """
    def __init__(self):
        self.test_results = []  # Stores trust scores from each test

    def run_test(self):
        """Executes a single test cycle until the desired probability is achieved."""
        test_results = []
        used_nodes = set()  # Track nodes already used in this test
        while True:
            selected_node = self._select_node(used_nodes)  # Select a node based on weights
            correct = np.random.rand() < selected_node.accuracy_score  # Simulate the test result
            selected_node.pick(correct)  # Update node based on the test result
            test_results.append(selected_node.trust_score)
            used_nodes.add(selected_node)  # Mark node as used

            if self._calculate_probability(test_results):
                break  # Stop if desired probability is achieved

        return test_results, len(test_results)

    def _select_node(self, used_nodes):
        """Selects an active node for testing, excluding already used nodes."""
        active_nodes = [node for node in oracle_nodes if node.is_active() and node not in used_nodes]
        if not active_nodes:
            raise Exception("No active oracle nodes available.")

        weights = [node.weight() for node in active_nodes]
        probabilities = [weight / sum(weights) for weight in weights]
        return np.random.choice(active_nodes, p=probabilities)

    def _calculate_probability(self, test_results):
        """Calculates the confidence interval to determine if the system is reliable."""
        total_tests = len(test_results)
        proportion_correct = np.mean(test_results)
        z_score = st.norm.ppf(1 - (1 - DESIRED_PROBABILITY) / 2)  # Z-score for the desired confidence level
        std_error = np.sqrt(proportion_correct * (1 - proportion_correct) / total_tests)
        margin_of_error = z_score * std_error
        return (proportion_correct - margin_of_error) > 0.5  # Check if lower bound > 0.5

class OracleBatchTest:
    """
    Conducts multiple test batches and aggregates metrics such as false positives and negatives.
    """
    def __init__(self, num_batches=100):
        self.num_batches = num_batches
        self.results = {
            'num_tests': [],
            'false_positives': [],
            'false_negatives': [],
            'incompetent_kicks': []
        }

    def run_batches(self):
        """Executes multiple test batches and records metrics."""
        for _ in range(self.num_batches):
            oracle_test = OracleTest()
            _, num_tests = oracle_test.run_test()
            self.results['num_tests'].append(num_tests)

            # Metrics calculation
            false_positives = sum(1 for node in oracle_nodes if node.type == 'competent' and not node.is_active())
            false_negatives = sum(1 for node in oracle_nodes if node.type != 'competent' and node.is_active())
            incompetent_kicks = sum(1 for node in oracle_nodes if node.type == 'incompetent' and not node.is_active())

            self.results['false_positives'].append(false_positives)
            self.results['false_negatives'].append(false_negatives)
            self.results['incompetent_kicks'].append(incompetent_kicks)

    def plot_metrics(self):
        """Generates and saves plots for the collected metrics."""
        df = pd.DataFrame(self.results)
        df['test_number'] = df.index + 1  # Add test number for x-axis

        metrics = [
            ('num_tests', 'Number of Tests to Achieve Desired Probability', 'num_tests_over_time.png'),
            ('false_positives', 'False Positives Over Time', 'false_positives_over_time.png'),
            ('false_negatives', 'False Negatives Over Time', 'false_negatives_over_time.png'),
            ('incompetent_kicks', 'Incompetent Nodes Kicked Out Over Time', 'incompetent_kicks_over_time.png')
        ]

        for metric, title, filename in metrics:
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='test_number', y=metric, data=df)
            plt.title(title)
            plt.xlabel('Test Number')
            plt.ylabel(metric.replace('_', ' ').capitalize())
            plt.savefig(os.path.join(OUTPUT_DIR, filename))
            plt.savefig(os.path.join(OUTPUT_DIR, filename.replace(".png", ".eps")))
            plt.close()

# Example usage
batch_test = OracleBatchTest(num_batches=1000)
batch_test.run_batches()
batch_test.plot_metrics()
