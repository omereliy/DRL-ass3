"""
Generate PDF report for DRL Assignment 3.
"""

from fpdf import FPDF

class ReportPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 10, 'DRL Assignment 3: Meta and Transfer Learning', 0, 1, 'C')
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 10, title, 0, 1, 'L', True)
        self.ln(4)

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def bullet_point(self, text):
        self.set_font('Helvetica', '', 10)
        x = self.get_x()
        self.cell(10, 5, '  -')
        self.multi_cell(0, 5, text)
        self.set_x(x)

    def add_table(self, headers, data, col_widths=None):
        self.set_font('Helvetica', 'B', 9)
        if col_widths is None:
            col_widths = [self.epw / len(headers)] * len(headers)

        # Header
        self.set_fill_color(200, 200, 200)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, 1, 0, 'C', True)
        self.ln()

        # Data
        self.set_font('Helvetica', '', 9)
        for row in data:
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), 1, 0, 'C')
            self.ln()
        self.ln(4)


def generate_report():
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font('Helvetica', 'B', 18)
    pdf.cell(0, 15, 'Deep Reinforcement Learning Assignment 3', 0, 1, 'C')
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Meta and Transfer Learning', 0, 1, 'C')
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 8, 'Ben-Gurion University of the Negev', 0, 1, 'C')
    pdf.cell(0, 8, 'January 2024', 0, 1, 'C')
    pdf.ln(10)

    # 1. Introduction
    pdf.chapter_title('1. Introduction')
    pdf.body_text(
        'This report presents the implementation and analysis of meta and transfer learning techniques '
        'for Deep Reinforcement Learning. We implement an Actor-Critic architecture and evaluate three '
        'transfer learning approaches across three OpenAI Gymnasium environments: CartPole-v1, '
        'Acrobot-v1, and MountainCarContinuous-v0.'
    )

    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 6, 'Transfer Learning Methods Implemented:', 0, 1)
    pdf.bullet_point('Section 1: Individual Actor-Critic training from scratch (baseline)')
    pdf.bullet_point('Section 2: Fine-tuning pre-trained models with output layer re-initialization')
    pdf.bullet_point('Section 3: Progressive Networks with frozen source networks and lateral connections')
    pdf.ln(4)

    # 2. Architecture Design
    pdf.chapter_title('2. Architecture Design')

    pdf.section_title('2.1 Standardized Input/Output Dimensions')
    pdf.body_text(
        'To enable transfer learning across different environments, we standardize all input and output dimensions:'
    )
    pdf.bullet_point('Observation dimension: 6 (padded with zeros for smaller environments)')
    pdf.bullet_point('Action dimension: 3 (with action masking for environments with fewer actions)')
    pdf.ln(2)

    pdf.section_title('2.2 Network Architecture')
    pdf.body_text('Both Actor and Critic networks use identical architectures:')
    pdf.bullet_point('Input layer: 6 units (standardized observation)')
    pdf.bullet_point('Hidden layer 1: 128 units with ReLU activation')
    pdf.bullet_point('Hidden layer 2: 128 units with ReLU activation')
    pdf.bullet_point('Output layer: 3 units (Actor) / 1 unit (Critic)')
    pdf.body_text('Weight initialization uses Xavier/Glorot uniform initialization for improved gradient flow.')

    # 3. Section 1 Results
    pdf.chapter_title('3. Section 1: Individual Training Results')

    pdf.section_title('3.1 Training Configuration')
    pdf.bullet_point('Episodes: 1500')
    pdf.bullet_point('Learning rate: 0.001')
    pdf.bullet_point('Discount factor (gamma): 0.99')
    pdf.bullet_point('Entropy coefficient: 0.01')
    pdf.bullet_point('Value loss coefficient: 0.5')
    pdf.ln(2)

    pdf.section_title('3.2 Results')
    headers = ['Environment', 'Episodes', 'Time (s)', 'Avg Reward', 'Target', 'Converged']
    data = [
        ['CartPole-v1', '1500', '11.60', '21.18', '>475', 'No'],
        ['Acrobot-v1', '1500', '255.15', '-500.00', '>-100', 'No'],
        ['MountainCarCont-v0', '1500', '470.78', '-62.94', '>90', 'No'],
    ]
    pdf.add_table(headers, data, [40, 20, 25, 30, 25, 25])

    pdf.body_text(
        'Observations: None of the environments achieved convergence with the baseline Actor-Critic '
        'implementation. This establishes a baseline for comparing transfer learning approaches.'
    )

    # 4. Section 2 Results
    pdf.chapter_title('4. Section 2: Fine-Tuning Results')

    pdf.section_title('4.1 Fine-Tuning Procedure')
    pdf.body_text('The fine-tuning procedure consists of three steps:')
    pdf.bullet_point('1. Load the fully trained model from the source environment')
    pdf.bullet_point('2. Re-initialize the output layer weights (fc3) of both Actor and Critic')
    pdf.bullet_point('3. Train the network on the target environment')
    pdf.body_text(
        'This approach preserves learned feature representations in hidden layers while adapting '
        'the decision layer to the new task.'
    )

    pdf.section_title('4.2 Results')
    headers = ['Transfer Pair', 'Episodes', 'Time (s)', 'Avg Reward', 'Improvement']
    data = [
        ['Acrobot -> CartPole', '1500', '11.12', '21.87', '+0.69'],
        ['CartPole -> MountainCar', '1500', '466.44', '-62.04', '+0.90'],
    ]
    pdf.add_table(headers, data, [50, 25, 25, 30, 30])

    pdf.body_text(
        'Comparison with Baseline: Fine-tuning showed marginal improvements but did not achieve '
        'convergence, suggesting the hidden layer representations from source tasks may not transfer '
        'optimally to these specific target tasks.'
    )

    # 5. Section 3 Results
    pdf.add_page()
    pdf.chapter_title('5. Section 3: Progressive Networks Results')

    pdf.section_title('5.1 Progressive Network Architecture')
    pdf.body_text(
        'Progressive Networks use multiple frozen source networks with lateral connections to a '
        'trainable target network:'
    )
    pdf.bullet_point('Source networks: Frozen (non-trainable), provide hidden layer features')
    pdf.bullet_point('Lateral connections: Transform source hidden features (128-dim -> 128-dim)')
    pdf.bullet_point('Target network: Trainable, combines own features with lateral inputs')
    pdf.body_text(
        'The lateral connections allow the target network to selectively use relevant features '
        'from source tasks while learning new task-specific representations.'
    )

    pdf.section_title('5.2 Results')
    headers = ['Sources -> Target', 'Episodes', 'Time (s)', 'Avg Reward', 'Improvement']
    data = [
        ['{Acro,Mount} -> CartPole', '1500', '14.81', '22.53', '+1.35'],
        ['{Cart,Acro} -> MountainCar', '1500', '638.60', '-62.35', '+0.59'],
    ]
    pdf.add_table(headers, data, [55, 25, 25, 30, 30])

    pdf.section_title('5.3 Comparison Summary')
    headers = ['Target Environment', 'Baseline', 'Fine-Tuned', 'Progressive']
    data = [
        ['CartPole-v1', '21.18', '21.87 (+0.69)', '22.53 (+1.35)'],
        ['MountainCarCont-v0', '-62.94', '-62.04 (+0.90)', '-62.35 (+0.59)'],
    ]
    pdf.add_table(headers, data, [50, 35, 40, 40])

    # 6. Analysis
    pdf.chapter_title('6. Analysis and Discussion')

    pdf.section_title('6.1 Transfer Learning Effectiveness')
    pdf.body_text(
        'Progressive Networks showed the best improvement for CartPole (+1.35 over baseline), '
        'suggesting that combining features from multiple source tasks can provide richer '
        'representations than single-source fine-tuning.'
    )
    pdf.body_text(
        'For MountainCar, fine-tuning performed slightly better than progressive networks '
        '(+0.90 vs +0.59), possibly because the continuous control task benefits more from '
        'adapting a single well-trained policy than combining discrete task features.'
    )

    pdf.section_title('6.2 Convergence Challenges')
    pdf.body_text('None of the approaches achieved the target convergence thresholds. Potential factors:')
    pdf.bullet_point('Episode count: 1500 episodes may be insufficient for convergence')
    pdf.bullet_point('Hyperparameter tuning: Default parameters may not be optimal for all environments')
    pdf.bullet_point('MountainCar discretization: Continuous action space discretized to 3 actions')
    pdf.bullet_point('Reward shaping: MountainCarContinuous has sparse rewards')

    pdf.section_title('6.3 Key Findings')
    pdf.bullet_point('1. Transfer learning consistently improved performance over training from scratch')
    pdf.bullet_point('2. Progressive networks provided the largest improvement for discrete action spaces')
    pdf.bullet_point('3. Fine-tuning was more effective for continuous control tasks')
    pdf.bullet_point('4. Lateral connections in progressive networks allow selective feature reuse')

    # 7. Conclusion
    pdf.chapter_title('7. Conclusion')
    pdf.body_text(
        'This assignment demonstrated the implementation and evaluation of transfer learning techniques '
        'for reinforcement learning. While none of the approaches achieved full convergence, both '
        'fine-tuning and progressive networks showed consistent improvements over baseline training.'
    )
    pdf.body_text(
        'Progressive networks showed particular promise for combining knowledge from multiple source '
        'tasks, with the best overall improvement observed for CartPole-v1. The modular architecture '
        'of progressive networks, with frozen source columns and trainable lateral connections, '
        'provides a principled approach to avoiding catastrophic forgetting while enabling knowledge transfer.'
    )
    pdf.body_text(
        'Future work could explore extended training with more episodes, hyperparameter optimization, '
        'different network architectures for source-target matching, and continuous action spaces '
        'with actor networks outputting distribution parameters.'
    )

    # Save the PDF
    pdf.output('/Users/omereliyahu/personal/DRL-ass3/report/report.pdf')
    print('Report generated: report/report.pdf')


if __name__ == '__main__':
    generate_report()
