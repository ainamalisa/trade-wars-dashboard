# TradeWars: Impact Analysis of US-China Tariffs on Global Trade and Economy

A Streamlit dashboard application for analyzing the impact of recent US-China tariffs on trade volumes and key economic indicators through comprehensive data processing, correlation analysis, and predictive modeling.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Data](#data)
* [Analysis Details](#analysis-details)
* [Technologies](#technologies)
* [Customization](#customization)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

---

## Project Overview

TradeWars is a data-driven project aimed at understanding the economic impact of tariffs imposed between the US and China, with a focus on how tariffs correlate with various economic indicators such as GDP growth, inflation rate, stock market changes, and employment rates across different countries.

This dashboard offers:

* Interactive navigation through multiple sections for clear insights.
* Clean and optimized sidebar UI with full-width buttons.
* Data preprocessing including handling missing data and filtering.
* Correlation analysis visualized with both bar plots and refined heatmaps using Plotly.
* Predictive modeling frameworks (future implementation).

---

## Features

* **Custom Sidebar Navigation:**
  Buttons without borders, full-width and left-aligned for easy access and compact layout without scrolling.

* **Data Overview:**
  Inspect raw and cleaned datasets, including preprocessing steps like replacing missing values and filtering years.

* **Trends in Trade Volume:**
  Visualize historical trade volume trends.

* **Sentiment Analysis:**
  Analyze sentiment trends from trade-related user reviews (planned or implemented).

* **Correlation Analysis:**
  Explore how tariff changes correlate with economic indicators by country, with interactive Plotly bar charts and heatmaps.

* **Predictive Modeling:**
  (Planned) Forecasting tariffs or economic outcomes based on historical data.

* **Conclusion & Recommendations:**
  Summarize insights and suggest actionable strategies.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/tradewars.git
   cd tradewars
   ```

2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

---

## Usage

* Navigate through the sidebar buttons to explore different analysis sections.
* Upload your dataset in the Data Overview section if applicable.
* Interact with charts for detailed insights on tariff impact.
* The sidebar is optimized to fit without scrolling, with compact spacing.

---

## Data

The project uses a curated dataset (`Dataset 3.csv`) containing:

* Country-level tariff data over years.
* Economic indicators such as GDP growth, inflation, stock indices, and employment.
* Data preprocessing includes:

  * Handling missing values (`..` replaced with NaN).
  * Filtering data for years outside 2015–2019.
  * Converting numeric fields for analysis.
  * Grouping data by country and filling missing values with medians.

---

## Analysis Details

### Correlation Analysis

* Calculate Pearson correlations between tariffs and economic indicators per country.
* Visualize:

  * Bar plots of correlations by country for each indicator.
  * Heatmap summarizing all correlations across countries and indicators.
* Interactive charts built with Plotly for enhanced user experience.

### Sidebar

* Implemented custom CSS for full-width, borderless sidebar buttons.
* Buttons are left-aligned with no scroll and minimal spacing between elements.

---

## Technologies

* Python 3.7+
* Streamlit for web app UI
* Pandas & NumPy for data processing
* Matplotlib, Seaborn for static plotting
* Plotly for interactive visualizations
* Scikit-learn for preprocessing and modeling

---

## Customization

* Modify the sidebar buttons and sections in `app.py` via the `sections` dictionary.
* Adjust CSS styles in the markdown injection block.
* Update data paths and preprocessing steps to match your dataset structure.
* Expand analysis sections or add new models as required.

---

## Contributing

Contributions are welcome! Feel free to:

* Report issues or bugs.
* Suggest new features or improvements.
* Submit pull requests with enhancements or fixes.

Please fork the repo and follow the standard GitHub workflow.
