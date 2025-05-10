<!-- #region -->
# Human Activity Monitoring Using Permutation Entropy and Complexity


### Author
**Rosalina Torres**  
Northeastern University  
IE6400 – Data Analytics Engineering  
Spring 2025  
Contact: torres.ros@northeastern.edu
🔗 [LinkedIn](#) | [GitHub](#)


### Project Overview
This project analyzes accelerometer data from human activities (walking, running, climbing up, climbing down) using permutation entropy and complexity metrics. By identifying optimal parameters for distinguishing between different physical activities, this work contributes to the rapidly growing field of human activity recognition, which has critical applications across healthcare, fitness, industrial safety, and smart environments.

## Dataset
The dataset contains accelerometer readings from 15 subjects performing four activities:
- Walking
- Running
- Climbing up
- Climbing down

Data was collected from chest-mounted sensors, capturing acceleration in the x, y, and z axes.

### Project Tasks

### Task 1: Load the Required Data
- Loading accelerometer data for all subjects and activities
- Processing and organizing the data for analysis

### Task 2: Compute Permutation Entropy and Complexity
Computing permutation entropy and complexity for various parameter combinations:
- Embedded Dimensions: 3, 4, 5, 6
- Embedded Delays: 1, 2, 3
- Signal Lengths: 1024, 2048, 4096

The calculations were performed for all subjects, activities, and accelerometer axes, resulting in 6480 rows of data.

### Task 3: Filter Data for a Specific Subject, Axis, and Signal Length
Selected Subject 3, x-axis accelerometer data for detailed analysis.

### Task 4: Identify Optimal Parameters for Walking vs. Running
Created scatter plots to identify the optimal dimension and delay for distinguishing walking from running activities.

**Findings:**
- Walking: Lower permutation entropy (~0.75) and higher complexity (~0.22)
- Running: Higher permutation entropy (~0.88) and lower complexity (~0.13)
- **Optimal parameters:** Dimension=5, Delay=2

### Task 5: Identify Optimal Parameters for Climbing Up vs. Climbing Down
Created scatter plots to identify the optimal dimension and delay for distinguishing climbing up from climbing down.

**Findings:**
- Climbing Up: Higher permutation entropy (~0.83) and lower complexity (~0.17)
- Climbing Down: Lower permutation entropy (~0.79) and higher complexity (~0.20)
- **Optimal parameters:** Dimension=4, Delay=3

## Industry Applications

### Healthcare and Remote Patient Monitoring
- **Early Disease Detection**: Changes in gait patterns can indicate neurodegenerative conditions like Parkinson's disease
- **Rehabilitation Monitoring**: Track patient recovery progress through quantitative movement analysis
- **Fall Detection and Prevention**: Identify high-risk movement patterns in elderly patients
- **Medication Effectiveness**: Measure improvements in mobility after treatment interventions

### Fitness and Sports Performance
- **Athlete Training Optimization**: Detailed analysis of movement patterns to improve technique
- **Injury Prevention**: Detect fatigue or improper form before injuries occur
- **Personalized Fitness Programs**: Tailor exercise recommendations based on movement quality
- **Performance Metrics**: Provide quantitative feedback on movement efficiency and consistency

### Industrial Safety and Ergonomics
- **Workplace Safety**: Monitor workers in dangerous environments for signs of fatigue
- **Ergonomic Assessment**: Identify movement patterns that may lead to repetitive strain injuries
- **Training Verification**: Ensure proper technique is being used for safety-critical tasks
- **Productivity Enhancement**: Optimize movement patterns for efficiency while maintaining safety

### Smart Environments and IoT
- **Smart Homes**: Automatically adjust lighting, temperature, or security based on detected activities
- **Energy Efficiency**: Power management based on occupant activities
- **Ambient Assisted Living**: Support independent living for elderly or disabled individuals
- **Context-Aware Computing**: Enhance user experience by anticipating needs based on current activity

### Security and Authentication
- **Biometric Authentication**: Use unique movement patterns as a security feature
- **Behavioral Analysis**: Detect suspicious activities in secure environments
- **Continuous Authentication**: Verify user identity throughout system usage
- **Fraud Detection**: Identify unusual patterns of physical activity indicating unauthorized access

### Entertainment and Gaming
- **Motion-Based Gaming**: Create more responsive and immersive gaming experiences
- **Virtual Reality**: Improve motion tracking for realistic VR experiences
- **Extended Reality Applications**: Enhance user interaction in mixed reality environments
- **Gesture Recognition**: Enable more natural human-computer interaction
<!-- #endregion -->

## Key Functions

### Shannon Entropy
```python
def s_entropy(freq_list):
    '''
    Computes the Shannon entropy of a given frequency distribution.
    
    Parameters:
    freq_list (list): List of frequencies/probabilities
    
    Returns:
    float: Shannon entropy value
    '''
    # Remove zero frequencies which would cause log(0) errors
    freq_list = [f for f in freq_list if f != 0]
    # Calculate Shannon entropy: -sum(p * log(p))
    sh_entropy = -sum(f * np.log(f) for f in freq_list)
    return sh_entropy

## Key Functions

### Shannon Entropy
```
def s_entropy(freq_list):
    '''
    Computes the Shannon entropy of a given frequency distribution.
    
    Parameters:
    freq_list (list): List of frequencies/probabilities
    
    Returns:
    float: Shannon entropy value
    '''
    # Remove zero frequencies which would cause log(0) errors
    freq_list = [f for f in freq_list if f != 0]
    # Calculate Shannon entropy: -sum(p * log(p))
    sh_entropy = -sum(f * np.log(f) for f in freq_list)
    return sh_entropym
    
    ### Permutation Entropy

    def p_entropy(op):
    '''
    Computes the permutation entropy for a time series.
    
    Parameters:
    op (list): Ordinal pattern frequencies
    
    Returns:
    float: Normalized permutation entropy value (between 0 and 1)
    '''
    ordinal_pat = op
    # Maximum possible entropy for the given number of patterns
    max_entropy = np.log(len(ordinal_pat))
    # Convert counts to probabilities
    p = np.divide(np.array(ordinal_pat), float(sum(ordinal_pat)))
    # Calculate and normalize entropy
    return s_entropy(p) / max_entropy


Complexity (Jensen-Shannon Divergence)

def complexity(op):
    '''
    Computes the complexity of a time series defined as: 
    Comp_JS = Q_o * JSdivergence * PE 
    where Q_o is the normalizing constant, JSdivergence is Jensen-Shannon divergence,
    and PE is permutation entropy.
    
    Parameters:
    op (list): Ordinal pattern frequencies
    
    Returns:
    float: Complexity value
    '''
    # Calculate permutation entropy
    pe = p_entropy(op)
    
    # Calculate normalizing constant Q_0
    constant1 = (0.5 + ((1 - 0.5) / len(op))) * np.log(0.5 + ((1 - 0.5) / len(op)))
    constant2 = ((1 - 0.5) / len(op)) * np.log((1 - 0.5) / len(op)) * (len(op) - 1)
    constant3 = 0.5 * np.log(len(op))
    Q_o = -1 / (constant1 + constant2 + constant3)

    # Probability distribution for the ordinal pattern
    temp_op_prob = np.divide(op, sum(op))
    # Create a mixture distribution (between ordinal pattern and uniform)
    temp_op_prob2 = (0.5 * temp_op_prob) + (0.5 * (1 / len(op)))
    
    # Jensen-Shannon Divergence calculation
    JSdivergence = (s_entropy(temp_op_prob2) - 0.5 * s_entropy(temp_op_prob) - 0.5 * np.log(len(op)))
    
    # Final complexity calculation
    Comp_JS = Q_o * JSdivergence * pe
    return Comp_JS

### Requirements

Python 3.x
NumPy
Pandas
Matplotlib
Seaborn
ordpy

### Usage

Clone this repository
Install the required packages: pip install -r requirements.txt
Run the Jupyter notebook: jupyter notebook project2ipynb.ipynb

### Business Impact
The methods demonstrated in this project have significant commercial potential:

Wearable Technology Development: The accuracy improvements in activity classification can enhance the value proposition of fitness trackers and smartwatches.
Healthcare Cost Reduction: Early detection of mobility issues through sophisticated activity monitoring can reduce hospitalization costs by up to 30%.
Workplace Safety ROI: Industrial implementations of activity monitoring systems have shown 40-60% reductions in workplace injuries, representing significant insurance and productivity savings.
IoT Integration: These algorithms can be integrated into existing IoT ecosystems to add significant value to smart home and office solutions.
Data Monetization: Aggregated and anonymized movement pattern data is valuable for urban planning, retail space optimization, and population health management.

### Conclusion
This project demonstrates that permutation entropy and complexity are effective features for distinguishing between human activities based on accelerometer data. Different physical activities show characteristic patterns in the permutation entropy vs. complexity space, allowing for effective discrimination.
The optimal parameters for distinguishing between activities depend on the specific activities being compared:

Walking vs. Running: Dimension=5, Delay=2
Climbing Up vs. Climbing Down: Dimension=4, Delay=3

These findings have immediate applications across multiple industries, particularly in healthcare monitoring, fitness technology, workplace safety, and smart environments where accurate activity recognition can drive improved outcomes and create significant business value.

```python
import gdown

# Use the file ID in the download link
file_id = '10Sh6VMOxQUT_n-8yBdNLpuj3yTGX-3gl'
url = f'https://drive.google.com/uc?id={file_id}'

# Download the file to the local environment
gdown.download(url, 'processed_permutation_entropy_complexity.csv', quiet=False)
```

```python

```

```python

```

```python

```

```python

```

```python

```
