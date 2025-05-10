import os
import shutil

print("Generating static files for GitHub Pages...")

# Create the docs directory if it doesn't exist
if not os.path.exists('docs'):
    os.makedirs('docs')

# Generate a simple static HTML for now
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Democracy Index Clustering Analysis</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }
        h1 {
            color: #1e3a8a;
        }
        h2 {
            color: #64748b;
            font-weight: normal;
        }
        .footer {
            margin-top: 40px;
            font-size: 12px;
            color: #64748b;
            text-align: center;
        }
    </style>
</head>
<body>
    <h1>Democracy Index Clustering Analysis</h1>
    <h2>Machine Learning Discovers Political Regime Patterns</h2>
    
    <p>This interactive dashboard is currently under development.</p>
    <p>The dashboard will showcase how unsupervised machine learning algorithms can naturally discover regime patterns from democracy data.</p>
    
    <h3>Key Features (Coming Soon):</h3>
    <ul>
        <li>Country-by-country democratic dimension analysis</li>
        <li>Democracy clustering visualization</li>
        <li>Machine learning model insights</li>
        <li>Historical democracy trends</li>
    </ul>
    
    <div class="footer">
        <p>Data source: Economist Intelligence Unit Democracy Index</p>
        <p>Compiled by Rosalina Torres as part of democracy clustering analysis project</p>
        <p>© 2025 | Northeastern University</p>
    </div>
</body>
</html>
"""

# Write to index.html
with open('docs/index.html', 'w') as f:
    f.write(html_content)

# Copy assets if they exist
if os.path.exists('dashboard/assets'):
    if not os.path.exists('docs/assets'):
        os.makedirs('docs/assets')
    for file in os.listdir('dashboard/assets'):
        source_path = os.path.join('dashboard/assets', file)
        if os.path.isfile(source_path):
            shutil.copy(source_path, os.path.join('docs/assets', file))

print("Static files generated in the 'docs' directory.")
