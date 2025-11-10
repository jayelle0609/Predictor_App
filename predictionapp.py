import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

pred_img = Image.open("pred.png")

st.image(pred_img, caption="Supervised Learning", use_column_width=True)
st.caption("**This app was created solely for an interview with Prime Ministers Office on 12-Nov-2025, for a research analytics position.**")

st.set_page_config(layout="centered")
st.title("Prediction App using Regression")
st.write("##### Predict a numerical target variable with regression models and visualize the results!")

# -------------------------------
# Upload CSV or use sample
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using default sample dataset (that Jia Ling made)")
    df = pd.read_csv("sample.csv")  # Ensure sample.csv exists

st.write("Data preview:")
st.dataframe(df.head())


# -------------------------------
# Drop unwanted columns
# -------------------------------
all_columns = df.columns.tolist()
cols_to_drop = st.multiselect("Select columns to drop (ex. irrelevant)", all_columns)
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    st.write(f"Dropped columns: {cols_to_drop}")

# -------------------------------
# Select target column 
# -------------------------------
target_col = st.selectbox("Select target column to predict", df.columns.tolist())
X = df.drop(columns=[target_col])
y = df[target_col]

st.write(f"Target: **{target_col}**")
st.write("Features:", X.columns.tolist())

# -------------------------------
# Identify numeric/categorical columns
# -------------------------------
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# -------------------------------
# Specify ordinal columns
# -------------------------------
st.write("### Specify Ordinal Columns")
st.caption("Ordinal columns are categorical columns where order matters.")
ordinal_cols = st.multiselect("Select ordinal columns", categorical_cols)

ordinal_orders = {}
for col in ordinal_cols:
    X[col] = X[col].astype(str).str.lower()
    unique_vals = X[col].unique()
    st.write(f"Unique values for '{col}': {unique_vals}")
    order_str = st.text_input(f"Enter order for **{col}** (comma-separated)", value=",".join(unique_vals))
    ordinal_orders[col] = [x.strip().lower() for x in order_str.split(",")]

categorical_cols = [c for c in categorical_cols if c not in ordinal_cols]

# -------------------------------
# Preprocessing info
# -------------------------------
st.caption("[EASY LAYMAN EXPLANATION OF TECHNICAL TERMS](https://docs.google.com/document/d/13CilRy_dplJYhaXDSUkflicZhsNIM_j5voAo74__Y2o/edit?usp=sharing)")

st.subheader("Backend Pipeline Workflow")
st.write("In the backend, the following preprocessing and analysis steps are performed:")
st.info("""
**Preprocessing the data:**
- Dropping any columns you chose.
- Imputing missing values:
  - Numeric columns: replaced with **mean** from training set.
  - Categorical columns: replaced with **mode** from training set.
- Scaling numeric features using **StandardScaler** to standardize the range.
- Encoding categorical features using **OneHotEncoder**.
- Encoding ordinal columns using **OrdinalEncoder** using the specified order.
""")

st.info("""
**Preparing data for modeling:**
- Dataset split into **training** and **validation** sets (80-20).
- Baseline regression models are trained and evaluated.
- **Cross-validation (k-fold)** applied to assess model performance and mean RMSE and R2 is calculated from each fold.
- Metrics calculated: 
  - **R²**: measures how well the model explains variance in the target variable (higher is better).
  - **RMSE**: measures the average prediction error magnitude (lower is better).
""")

st.info("""
**Hyperparameter tuning of best model:**
- Selection of best baseline model : **highest R² and lowest RMSE**.
- **RandomizedSearchCV** explores multiple combinations of hyperparameters.
- **Best hyperparameters** are fitted into model to **make prediction on unseen data**.
- Unseen data will be pre-processed with the pipeline before generating predictions. 
""")
# -------------------------------
# User selects sample fraction and folds
# -------------------------------
sample_frac = st.slider("Fraction of dataset to use", 0.1, 1.0, 1.0, 0.05)
kf_num = st.number_input("Number of Cross Validation folds", min_value=2, max_value=10, value=3, step=1)


X_sample = X.sample(frac=sample_frac, random_state=42)
y_sample = y.loc[X_sample.index]

# -------------------------------
# Preprocessing pipelines
# -------------------------------
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
])

ordinal_pipeline = None
if ordinal_cols:
    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=[ordinal_orders[c] for c in ordinal_cols]))
    ])

transformers = [('num', numeric_pipeline, numeric_cols),
                ('cat', categorical_pipeline, categorical_cols)]
if ordinal_cols:
    transformers.append(('ord', ordinal_pipeline, ordinal_cols))
preprocessor = ColumnTransformer(transformers=transformers)

# -------------------------------
# Regression models
# -------------------------------
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'CatBoost': CatBoostRegressor(iterations=100, learning_rate=0.1, random_seed=42, verbose=0)
}

# -------------------------------
# Train/validation split
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42
)

# -------------------------------
# Evaluate baseline models
# -------------------------------
cv_results = []
for name, model in models.items():
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2_val = r2_score(y_val, y_val_pred)
    cv_results.append({'Model': name, 'R2': r2_val, 'RMSE': rmse_val})

cv_df = pd.DataFrame(cv_results)
cv_df['rank'] = cv_df['R2'].rank(ascending=False) + cv_df['RMSE'].rank(ascending=True)
best_idx = cv_df['rank'].idxmin()
cv_df_sorted = cv_df.sort_values(by='R2', ascending=False)

def highlight_best(row):
    return ['background-color: #90ee90' if row.name == best_idx else '']*len(row)

st.caption("Higher R2 and lower RMSE reflects better model accuracy.")
st.write("# Model Evaluation Metrics")
st.dataframe(cv_df_sorted.drop(columns='rank').style.apply(highlight_best, axis=1))

best_model_name = cv_df.loc[best_idx, 'Model']
st.caption(f"Best Model : **{best_model_name}**")

# -------------------------------
# Bar chart
# -------------------------------
r2_colors = cv_df_sorted['R2']
rmse_colors_scaled = cv_df_sorted['RMSE'].max() - cv_df_sorted['RMSE']

fig = make_subplots(rows=1, cols=2, subplot_titles=("R2 of Baseline Models", "RMSE of Baseline Models"))
fig.add_trace(go.Bar(x=cv_df_sorted['Model'], y=cv_df_sorted['R2'],
                     text=np.round(cv_df_sorted['R2'],3), textposition='auto',
                     marker=dict(color=r2_colors, colorscale='magma_r', showscale=False)), row=1, col=1)
fig.add_trace(go.Bar(x=cv_df_sorted['Model'], y=cv_df_sorted['RMSE'],
                     text=np.round(cv_df_sorted['RMSE'],3), textposition='auto',
                     marker=dict(color=rmse_colors_scaled, colorscale='magma_r', showscale=False)), row=1, col=2)
fig.update_layout(title_text="Performance Metrics of Baseline Models", showlegend=False, width=1000, height=500)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Hyperparameter tuning grids
# -------------------------------
def get_param_grid(name):
    if name in ['XGBoost','LightGBM','CatBoost']:
        return {
            'model__n_estimators': np.arange(200,1000,100),
            'model__max_depth': np.arange(3,10),
            'model__learning_rate': np.linspace(0.01,0.3,5),
            'model__subsample': np.linspace(0.6,1.0,5),
            'model__colsample_bytree': np.linspace(0.6,1.0,5),
            'model__gamma':[0,1,2],
            'model__reg_alpha':[0,0.5,1],
            'model__reg_lambda':[1,2,3]
        }
    elif name=='Gradient Boosting':
        return {
            'model__n_estimators':[100,200,300,500],
            'model__learning_rate':[0.01,0.05,0.1,0.2],
            'model__max_depth':[2,3,4,5],
            'model__min_samples_split':[2,5,10],
            'model__min_samples_leaf':[1,2,4],
            'model__subsample':[0.6,0.8,1.0],
            'model__max_features':['sqrt','log2',None],
            'model__loss':['squared_error','absolute_error','huber']
        }
    elif name=='Random Forest':
        return {
            'model__n_estimators':[100,300,500,800],
            'model__max_depth':[None,5,10,15],
            'model__min_samples_split':[2,5,10],
            'model__min_samples_leaf':[1,2,4],
            'model__max_features':['sqrt','log2',None],
            'model__bootstrap':[True,False]
        }
    elif name in ['Ridge Regression','Lasso Regression']:
        return {
            'model__alpha': np.logspace(-3,2,6),
            'model__fit_intercept':[True,False],
            'model__copy_X':[True]
        }
    elif name == 'Linear Regression':
        return {
            'model__fit_intercept': [True, False],
            'model__copy_X': [True],
            'model__positive': [True, False]
        }
    else:
        return {}

# -------------------------------
# Model tuning section
# -------------------------------
model_names = list(models.keys())
best_model_idx = model_names.index(best_model_name)
selected_model_name = st.selectbox("Select a model to run hyperparameter tuning",
                                   model_names, index=best_model_idx)
selected_model = models[selected_model_name]

pipeline = Pipeline([('preprocessor', preprocessor), ('model', selected_model)])
param_dist = get_param_grid(selected_model_name)

if param_dist:
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=30,
                                       scoring='neg_root_mean_squared_error', cv=kf_num,
                                       n_jobs=-1, random_state=42, verbose=2)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    st.write("### Best Hyperparameters")
    st.write(random_search.best_params_)
else:
    pipeline.fit(X_train, y_train)
    best_model = pipeline

# -------------------------------
# Comparison Table 
# -------------------------------
baseline_model_pipeline = Pipeline([('preprocessor', preprocessor), ('model', models[best_model_name])])
baseline_model_pipeline.fit(X_train, y_train)
y_val_pred_baseline = baseline_model_pipeline.predict(X_val)
r2_baseline = r2_score(y_val, y_val_pred_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_val, y_val_pred_baseline))

y_val_pred_tuned = best_model.predict(X_val)
r2_tuned = r2_score(y_val, y_val_pred_tuned)
rmse_tuned = np.sqrt(mean_squared_error(y_val, y_val_pred_tuned))

comparison_df = pd.DataFrame({
    'Status': ['Before Tuning', 'After Tuning'],
    'R2': [r2_baseline, r2_tuned],
    'RMSE': [rmse_baseline, rmse_tuned]
})

st.write(f"## {best_model_name} Metrics:")

def highlight_after(row):
    color = 'background-color: #90ee90' if row['Status'] == 'After Tuning' else ''
    return [color]*len(row)

st.dataframe(
    comparison_df.style.format({'R2': "{:.3f}", 'RMSE': "{:.3f}"}).apply(highlight_after, axis=1)
)

# -------------------------------
# Feature coef
# -------------------------------
st.write(f"## Feature Contributions & Predicted vs Actual for {selected_model_name}")

preprocessor.fit(X_train)
num_features = numeric_cols
cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols) if categorical_cols else []
ord_features = ordinal_cols if ordinal_cols else []
all_features = np.concatenate([num_features, cat_features, ord_features])

# Create subplot: 1 row, 2 columns
fig_combined = make_subplots(rows=1, cols=2, subplot_titles=("Feature Coefficients", "Predicted vs Actual"))

# Left: Feature contributions
if selected_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
    coefs = best_model.named_steps['model'].coef_
    intercept = best_model.named_steps['model'].intercept_
    coef_df = pd.DataFrame({'Feature': all_features, 'Coefficient': coefs})
    
    # Bar chart
    fig_combined.add_trace(go.Bar(x=coef_df['Feature'], y=coef_df['Coefficient'], marker_color='royalblue', name = "Feature Coefficients"), row=1, col=1)
    
    # Equation string
    equation_terms = [f"{coef:.2f}*{feat}" for coef, feat in zip(coefs, all_features)]
    equation = f"{target_col} = {intercept:.2f} + " + " + ".join(equation_terms)
    st.write("### Regression Equation:")
    st.code(equation)

else:  # Tree models: feature importances
    importances = best_model.named_steps['model'].feature_importances_
    coef_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
    fig_combined.add_trace(go.Bar(x=coef_df['Feature'], y=coef_df['Importance'], marker_color='orange'), row=1, col=1)

# Right: Predicted vs Actual
y_train_pred = best_model.predict(X_train)
y_val_pred = best_model.predict(X_val)
fig_combined.add_trace(go.Scatter(x=y_train, y=y_train_pred, mode='markers', name='Train'), row=1, col=2)
fig_combined.add_trace(go.Scatter(x=y_val, y=y_val_pred, mode='markers', name='Validation'), row=1, col=2)
fig_combined.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode='lines', line=dict(color='red', dash='dash'), name='45-degree line'), row=1, col=2)

fig_combined.update_xaxes(title_text="Feature", row=1, col=1)
fig_combined.update_yaxes(title_text="Coefficient Value", row=1, col=1)
fig_combined.update_xaxes(title_text="Actual Values", row=1, col=2)
fig_combined.update_yaxes(title_text="Predicted Values", row=1, col=2)
fig_combined.update_layout(showlegend=True, height=500, width=1000)
st.plotly_chart(fig_combined, use_container_width=True)

# -------------------------------
# Forecasting / Prediction Section
# -------------------------------
st.write("# Forecasting Predictions on Unseen Data")
st.caption("Provide data for predictions: manual input or upload your CSV. If no file is uploaded, default test.csv will be used.")

# Input choice
input_choice = st.radio(
    "Select input method",
    options=["Manual Input", "Upload CSV"]
)

# Select model for prediction
model_choice = st.radio(
    "Select model for prediction",
    options=[f"Baseline: {best_model_name}", "After Hyperparameter Tuning"]
)

# Define selected model pipeline
if "Baseline" in model_choice:
    selected_pred_model = Pipeline([('preprocessor', preprocessor), ('model', models[best_model_name])])
    selected_pred_model.fit(X_train, y_train)
else:
    selected_pred_model = best_model

# Prepare test_df
test_df = pd.DataFrame()

if input_choice == "Manual Input":
    manual_data = {}
    numeric_means = X_train[numeric_cols].mean()
    categorical_modes = X_train[categorical_cols + ordinal_cols].mode().iloc[0] if (categorical_cols + ordinal_cols) else pd.Series()
    
    for col in X.columns:
        if col in numeric_cols:
            val = st.text_input(f"Input value for {col}", value=str(round(numeric_means[col],2)))
            manual_data[col] = [float(val)]
        else:
            unique_vals = X[col].unique().tolist()
            default_val = categorical_modes[col] if col in categorical_modes else unique_vals[0]
            val = st.selectbox(f"Select value for {col}", options=unique_vals, index=unique_vals.index(default_val))
            manual_data[col] = [val]
    test_df = pd.DataFrame(manual_data)

elif input_choice == "Upload CSV":
    uploaded_test = st.file_uploader("Upload Test CSV", type=["csv"])
    if uploaded_test:
        test_df = pd.read_csv(uploaded_test)
    else:
        # Use default test.csv if no upload
        test_df = pd.read_csv("test.csv")  
        st.info("Using default sample dataset (that Jia Ling made)")
# Fill missing values for numeric and categorical columns
for col in X.columns:
    if col not in test_df.columns:
        test_df[col] = np.nan  # ensure column exists
    if col in numeric_cols:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(X_train[col].mean())
    else:
        test_df[col] = test_df[col].fillna(X_train[col].mode()[0])
        test_df[col] = test_df[col].astype(str).str.lower()  # lowercase to match training

# Drop columns the user dropped during training
if cols_to_drop:
    test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])

# Predict button
if st.button("Predict"):
    if not test_df.empty:
        try:
            # Pass through pipeline (preprocessing + model)
            y_test_pred = selected_pred_model.predict(test_df)
            test_df[f'Predicted {target_col}'] = y_test_pred
            st.write(test_df[[f'Predicted {target_col}']])

            # Download button
            csv_bytes = test_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.warning("No test data available. Please provide input manually or upload a CSV.")

##################################################################################
st.markdown("---")
st.title("Check Out Jayelle's Portfolio!")

st.markdown("""
Welcome! Here are some of my personal websites and portfolio pages where you can learn more about me and my work:
""")

# List of links
links = {
    "My Website": "https://jayelle0609.github.io/jialing",
    "Tableau Visualizations": "https://public.tableau.com/app/profile/jialingteo/vizzes",
    "GitHub Portfolio": "https://github.com/jayelle0609/Portfolio",
    "Linkedin" : "https://www.linkedin.com/in/jialingteo/",
    "KMeans Clustering App (for PMO interview)" : "https://jialingkmeans.streamlit.app"

}

for name, url in links.items():
    st.markdown(f"- [{name}]({url})")

st.markdown("""
---
*Feel free to reach out or explore more!*  
<span style="font-size:10px;">
[Email Me!](mailto:jayelleteo@gmail.com) | [WhatsApp Me!](https://wa.me/6580402496)
</span>
<br>
<span style="font-size:12px; color:gray;">

</span>
""", unsafe_allow_html=True)
