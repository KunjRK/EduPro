import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Set page configuration layout
st.set_page_config(page_title="EduPro Analytics Dashboard", layout="wide")

# --- 1. DATA CACHING & PIPELINE ---
@st.cache_data
def load_and_prep_data():
    try:
        # Load sheets from the working project path directory
        courses = pd.read_csv('EduPro Online Platform.xlsx - Courses.csv')
        transactions = pd.read_csv('EduPro Online Platform.xlsx - Transactions.csv')
        teachers = pd.read_csv('EduPro Online Platform.xlsx - Teachers.csv')
    except FileNotFoundError:
        st.error("Please ensure your EduPro CSV files are in the exact same directory as app.py.")
        st.stop()

    # Create a 1-to-1 relationship mapping CourseID to a singular primary TeacherID from transactions
    course_teacher_map = transactions.groupby('CourseID')['TeacherID'].agg(lambda x: x.mode()[0]).reset_index()

    # Aggregate Transactions to Course Level to derive modeling target indicators
    course_stats = transactions.groupby('CourseID').agg(
        Enrollment_Count=('TransactionID', 'count'),
        Course_Revenue=('Amount', 'sum')
    ).reset_index()

    # Merge structures sequentially starting with Courses
    joined_df = pd.merge(courses, course_stats, on='CourseID', how='left').fillna(0)
    joined_df = pd.merge(joined_df, course_teacher_map, on='CourseID', how='left')
    joined_df = pd.merge(joined_df, teachers, on='TeacherID', how='left')
    
    # Feature Engineering: Verify if instructor expertise matches the category domain
    if 'Expertise' in joined_df.columns and 'CourseCategory' in joined_df.columns:
        joined_df['Expertise_Match'] = (joined_df['CourseCategory'].str.lower() == joined_df['Expertise'].str.lower()).astype(int)
    else:
        joined_df['Expertise_Match'] = 0
        
    return joined_df

# --- DEFINING GLOBAL DATAFRAME ---
# This ensures 'df' is correctly generated globally for the script elements below
df = load_and_prep_data()

# --- 2. MODEL TRAINING LAYER ---
@st.cache_resource
def train_models(df_input):
    num_features = ['CoursePrice', 'CourseDuration', 'CourseRating', 'YearsOfExperience', 'TeacherRating', 'Expertise_Match']
    cat_features = ['CourseCategory', 'CourseLevel', 'CourseType']
    
    X = df_input[num_features + cat_features]
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])
    
    # Model 1: Demand Estimation Machine
    model_enroll = Pipeline(steps=[('preprocessor', preprocessor), 
                                    ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
    model_enroll.fit(X, df_input['Enrollment_Count'])
    
    # Model 2: Course Revenue Engine
    model_rev = Pipeline(steps=[('preprocessor', preprocessor), 
                                 ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
    model_rev.fit(X, df_input['Course_Revenue'])
    
    return model_enroll, model_rev, num_features, cat_features

# Unpacking variables securely using our loaded global dataframe
model_enroll, model_rev, num_features, cat_features = train_models(df)

# --- 3. STREAMLIT INTERACTIVE SIDEBAR SIMULATOR ---
st.sidebar.header("🎯 Live Course Simulation Engine")

input_category = st.sidebar.selectbox("Course Category", options=df['CourseCategory'].unique())
input_level = st.sidebar.selectbox("Course Difficulty Level", options=df['CourseLevel'].unique())
input_type = st.sidebar.radio("Payment Matrix Type", options=df['CourseType'].unique())

input_price = st.sidebar.slider("Course Price ($)", min_value=0.0, max_value=float(df['CoursePrice'].max()), value=150.0)
input_duration = st.sidebar.slider("Course Duration (Hours)", min_value=1, max_value=int(df['CourseDuration'].max()), value=20)
input_rating = st.sidebar.slider("Target Student Rating", min_value=1.0, max_value=5.0, value=4.2, step=0.1)

input_exp = st.sidebar.slider("Instructor Profile Experience (Years)", min_value=0, max_value=int(df['YearsOfExperience'].max()), value=5)
input_t_rating = st.sidebar.slider("Instructor Rating Profile", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
input_exp_match = st.sidebar.checkbox("Instructor Specialization Matches Category?", value=True)

# Package interactive inputs for immediate engine prediction tracking
input_data = pd.DataFrame([{
    'CoursePrice': input_price,
    'CourseDuration': input_duration,
    'CourseRating': input_rating,
    'YearsOfExperience': input_exp,
    'TeacherRating': input_t_rating,
    'Expertise_Match': int(input_exp_match),
    'CourseCategory': input_category,
    'CourseLevel': input_level,
    'CourseType': input_type
}])

# Calculate targets dynamically via trained pipelines
pred_enrollment = max(0, int(model_enroll.predict(input_data)[0]))
pred_revenue = max(0.0, float(model_rev.predict(input_data)[0]))

# --- 4. STREAMLIT MAIN VIEWPORT PAGE ELEMENTS ---
st.title("🎓 EduPro Demand & Revenue Predictive Control Room")
st.markdown("Run simulation profiles on custom product criteria using trained structural machine learning models.")

# Summary Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Predicted Enrollment Count", f"{pred_enrollment:,} Signups")
with col2:
    st.metric("Predicted Course Revenue", f"${pred_revenue:,.2f}")
with col3:
    st.metric("Platform Active Catalog Size", f"{len(df)} Courses")
with col4:
    st.metric("Calculated Value Per Signup", f"${(pred_revenue / pred_enrollment if pred_enrollment > 0 else 0):,.2f}")

st.write("---")

# Navigation Interface
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Demand & Forecast Analytics", 
    "💰 Revenue Dynamic Trackers", 
    "🔍 Feature Importance Explorer", 
    "📊 Macro Category Comparisons"
])

with tab1:
    st.header("Course Demand Analysis Mapping")
    fig_demand = px.histogram(df, x="Enrollment_Count", nbins=20, title="Historical Catalog Enrollment Distribution Patterns",
                              labels={'Enrollment_Count': 'Enrollment Count'}, color_discrete_sequence=['#4A90E2'])
    fig_demand.add_vline(x=pred_enrollment, line_dash="dash", line_color="red", 
                         annotation_text=f"Simulated Target Point ({pred_enrollment})")
    st.plotly_chart(fig_demand, use_container_width=True)

with tab2:
    st.header("Revenue Forecast Deep-Dive Visualizations")
    fig_rev = px.scatter(df, x="CourseDuration", y="Course_Revenue", color="CourseCategory", size="CoursePrice",
                         title="Correlation Vector: Duration vs. Historical Revenue Generation Matrix",
                         labels={'CourseDuration': 'Duration (Hours)', 'Course_Revenue': 'Total Revenue ($)'})
    fig_rev.add_scatter(x=[input_duration], y=[pred_revenue], mode="markers", marker=dict(color="red", size=18, symbol="star"),
                        name="Your Current Simulated Profile")
    st.plotly_chart(fig_rev, use_container_width=True)

with tab3:
    st.header("Random Forest Multi-Dimensional Driver Analysis")
    ohe_features = list(model_rev.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_features))
    all_features_mapped = num_features + ohe_features
    feature_importances = model_rev.named_steps['model'].feature_importances_
    
    imp_df = pd.DataFrame({'Feature': all_features_mapped, 'Weight': feature_importances}).sort_values(by='Weight', ascending=False).head(10)
    
    fig_imp = px.bar(imp_df, x='Weight', y='Feature', orientation='h', title="Top 10 Computational Weights Influencing Revenue Predictions",
                     color='Weight', color_continuous_scale='Viridis')
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)

with tab4:
    st.header("Category-Level Performance Indices")
    cat_summary = df.groupby('CourseCategory').agg(
        Total_Enrollments=('Enrollment_Count', 'sum'),
        Total_Revenue=('Course_Revenue', 'sum'),
        Average_Course_Rating=('CourseRating', 'mean')
    ).reset_index()
    
    fig_cat = px.bar(cat_summary, x='CourseCategory', y='Total_Enrollments', color='Total_Revenue',
                     title="Aggregated Cross-Category Demand Volume vs Financial Performance Index",
                     labels={'Total_Enrollments': 'Total Signups Across Category', 'CourseCategory': 'Category Domain'},
                     color_continuous_scale='Cividis')
    st.plotly_chart(fig_cat, use_container_width=True)
    
    st.dataframe(cat_summary.style.format({
        'Total_Enrollments': '{:,.0f}',
        'Total_Revenue': '${:,.2f}',
        'Average_Course_Rating': '{:,.2f}'
    }), use_container_width=True)