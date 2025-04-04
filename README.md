# Healthcare_Insurance

This project performs end-to-end data analysis and predictive modeling on a healthcare insurance dataset to understand key factors influencing medical charges and build a machine learning model to estimate charges based on client attributes.

---

## 📁 Dataset

The dataset includes the following columns:

| Column     | Description                        |
|------------|------------------------------------|
| age        | Age of the primary beneficiary     |
| sex        | Gender (male/female)               |
| bmi        | Body Mass Index                    |
| children   | Number of children/dependents      |
| smoker     | Whether the person is a smoker     |
| region     | Region in the US                   |
| charges    | Medical insurance charges billed   |

---

## 📊 Exploratory Data Analysis (EDA)

# 1. Univariate Analysis

  - Distribution of `age`, `bmi`, `charges`, and `children`
  - Count plots for `sex`, `smoker`, `region`


a. Categorical Variables (Bar Plots)

1. Sex Distribution : sns.countplot(x='sex') was used.
                      Nearly equal number of males and females in the 
                      dataset.
<img width="424" alt="Image" src="https://github.com/user-attachments/assets/2b71323d-9877-46e8-a357-7ab9105c47e3" />


2. Smoker Status : Showed significantly fewer smokers than non-smokers.
                   Important because smokers are known to incur higher 
                   healthcare costs.
<img width="432" alt="Image" src="https://github.com/user-attachments/assets/ae722466-d393-44d0-81cb-ce812b04165d" />


3. Region : Balanced distribution across four US regions (southwest, 
            southeast, northwest, northeast).

<img width="422" alt="Image" src="https://github.com/user-attachments/assets/a7a36ffe-a4fb-44f5-9ba0-52609bb9459d" />

b. Numerical Variables (Histograms & Box Plots)

1. Age Distribution : Uniformly distributed from age 18 to mid-60s.
                      Suggests no skew toward younger or older populations.

2. BMI Distribution : Right-skewed slightly; most individuals are in the 
                      healthy to overweight range.
                      Box plot helped detect a few outliers indicating 
                      potential obesity.

3. Children : Most people have 0–2 children; fewer have 4–5.

4. Charges : Highly right-skewed. Most people have moderate charges, with a 
             few incurring very high costs (especially smokers or older 
             individuals).



# 2. Bivariate Analysis
These plots explored how insurance charges are affected by other variables.
 
  - Scatter plot of `age` vs `charges`, colored by smoking status
  - Box plots for `charges` by `region` and `children`
  

a. Scatter Plot: Age vs Charges (Colored by Smoker Status)
Code: sns.scatterplot(x='age', y='charges', hue='smoker')

<img width="439" alt="Image" src="https://github.com/user-attachments/assets/73eb92d4-643a-4a4e-854e-3aa89996a6fb" />

Insights:
  - Non-smokers: charges increase steadily with age.

  - Smokers: show a steep spike in charges regardless of age.

  - Smoking has a dramatic effect on increasing insurance costs.


b. Box Plot: Charges by Region
Code: sns.boxplot(x='region', y='charges')

Insights:

  - No significant regional difference in median charges.

  - Variability in charges is quite similar across regions.


c. Box Plot: Charges by Number of Children
Code: sns.boxplot(x='children', y='charges')

<img width="436" alt="Image" src="https://github.com/user-attachments/assets/f01b834d-3cb4-4680-a82a-09d14718e85e" />

Insights:

  - Slight increase in median charges with more children, but not a strong      correlation.

  - Variability remains similar across child counts.


# 3. Correlation Heatmap
  - Understand relationships between numeric variables

<img width="381" alt="Image" src="https://github.com/user-attachments/assets/741ed6a9-28c8-40af-b0e8-676e57351b92" />


Key Findings:

  - Strong positive correlation with age and BMI.

  - Moderate correlation with children.

# Categorical variables like smoker are encoded later and play a huge role (observed in model performance and feature importance)


# 4. Model Evaluation Visuals


a. Scatter Plot: Actual vs Predicted Charges
Used purple color for visual appeal: plt.scatter(y_test, y_pred_rf, color='purple')

<img width="504" alt="Image" src="https://github.com/user-attachments/assets/87576e3c-4756-4f32-9f9c-b6c0d7b4d415" />


Insight:

  - Shows how well the model predicted real-world charges.

  - Points close to the diagonal line indicate good model performance.


b. Feature Importance (Random Forest)

  - Smoker, age, and BMI were the top predictors of charges.

  - Region and number of children had less impact on the cost.



- **Insights**
  - Smokers have significantly higher charges
  - Charges tend to increase with age and BMI

---

## 🧪 Preprocessing

- Missing values check
- Encoding categorical variables using `pd.get_dummies()`
- Train-test split (80-20)

---

## 🤖 Machine Learning Models

We built and evaluated the following models:

- **Linear Regression**
- **Random Forest Regressor**
- *(Optional: XGBoost, Ridge, etc.)*

### Evaluation Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R-squared Score (R²)

### Visualizations

- Scatter plot: Predicted vs Actual charges (purple colored)
- Feature importance plot for Random Forest

---

## 📈 Key Results and Conclusion 

📌 Conclusions from the Analysis
Based on the EDA and machine learning results, here are the key conclusions:

1. Top Factors Influencing Insurance Charges
Smoking status is the most impactful feature. Smokers pay significantly more in healthcare insurance.

Age plays a major role: older individuals tend to have higher charges.

BMI also influences costs, likely due to associated health risks.

Children and region have a relatively minor effect.

2. Charges are Right-Skewed
Most individuals have moderate costs, while a few incur extremely high expenses — possibly due to smoking or chronic health conditions.

3. Model Performance
Random Forest Regressor gave an R² of ~0.88, outperforming Linear Regression (R² ~0.75), indicating strong predictive ability with non-linear relationships.

Models could be improved further with hyperparameter tuning or feature engineering (e.g., interaction terms like age × smoker).


| Model               | R² Score |
|---------------------|----------|
| Linear Regression   | ~0.75    |
| Random Forest       | ~0.88+   |

Random Forest outperformed linear regression due to its ability to capture non-linear patterns.

---

## 💡 Future Enhancements

- Hyperparameter tuning using GridSearchCV
- Try boosting models (e.g., XGBoost, LightGBM)
- Deploy the model using Flask or Streamlit
- Incorporate more healthcare-specific features if available

---
