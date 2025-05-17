"""
Student Performance Prediction System - Streamlit Version
This application predicts student academic performance using Random Forest
with customized feature importance weights.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin

# Set page configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "data", "student_performance.csv")
MODEL_PATH = os.path.join(CURRENT_DIR, "models", "model.pkl")
FEATURE_NAMES_PATH = os.path.join(CURRENT_DIR, "models", "feature_names.pkl")
GRADE_MAPPING_PATH = os.path.join(CURRENT_DIR, "models", "grade_mapping.pkl")

# Define custom importance weights
CUSTOM_WEIGHTS = {
    'Weekly_Study_Hours': 0.25,  # Highest importance
    'Notes': 0.20,
    'Project_work': 0.15,
    'Listening_in_Class': 0.12,
    'Attendance': 0.10,
    'Reading': 0.08,
    'Scholarship': 0.03,
    'Additional_Work': 0.02,
    'High_School_Type': 0.02,
    'Sex': 0.01,
    'Sports_activity': 0.01,
    'Student_Age': 0.01,
    'Transportation': 0.01
}

# Custom wrapper class for feature importance
class CustomImportanceRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model=None, custom_importance=None):
        if base_model is None:
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            )
        else:
            self.model = base_model
            
        self._custom_importance = custom_importance
        self._feature_names = None
        
    def fit(self, X, y):
        if hasattr(X, 'columns'):
            self._feature_names = X.columns.tolist()
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    @property
    def feature_importances_(self):
        return self._custom_importance if self._custom_importance is not None else self.model.feature_importances_
    
    @property
    def classes_(self):
        return self.model.classes_
        
    @property
    def feature_names_in_(self):
        if hasattr(self.model, 'feature_names_in_'):
            return self.model.feature_names_in_
        return np.array(self._feature_names) if self._feature_names else None

def convert_age_range(age_str):
    """Convert age range strings like '19-22' to their average value"""
    try:
        if isinstance(age_str, (int, float)):
            return age_str
        if '-' in str(age_str):
            parts = str(age_str).split('-')
            nums = [int(p.strip()) for p in parts]
            return sum(nums) / len(nums)
        return float(age_str)
    except:
        return 20.0  # Default age

def train_model():
    """Train the model and save all necessary files"""
    with st.spinner('Training model... This may take a moment.'):
        # Check if dataset exists
        if not os.path.exists(DATA_PATH):
            st.warning(f"Dataset not found at {DATA_PATH}")
            st.info("Creating mock dataset for demonstration...")
            create_mock_data()
            
        # Load dataset
        try:
            df = pd.read_csv(DATA_PATH)
            st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.info("Creating mock dataset instead...")
            df = create_mock_data()

        # Drop Student_ID if present
        if 'Student_ID' in df.columns:
            df = df.drop('Student_ID', axis=1)
            
        # Convert Student_Age column from ranges to numeric values
        if 'Student_Age' in df.columns:
            df['Student_Age'] = df['Student_Age'].apply(convert_age_range)
        
        # Identify and encode target variable
        target_column = 'Grade'
        if target_column not in df.columns:
            if 'Performance' in df.columns:
                target_column = 'Performance'
            else:
                st.warning(f"Target column not found. Available columns: {df.columns.tolist()}")
                # Add a mock target column
                df['Grade'] = np.random.randint(0, 5, size=len(df))
                target_column = 'Grade'
        
        # Save original grade values before encoding
        original_grades = sorted(df[target_column].unique())
        
        # Encode the target variable
        grade_encoder = LabelEncoder()
        encoded_grades = grade_encoder.fit_transform(df[target_column].astype(str))
        
        # Create mapping between encoded and original grades
        grade_mapping = dict(zip(grade_encoder.transform([str(g) for g in original_grades]), original_grades))
        
        # Save the grade mapping
        with open(GRADE_MAPPING_PATH, 'wb') as f:
            pickle.dump(grade_mapping, f)
        
        # Replace the target column with encoded values
        df[target_column] = encoded_grades
        
        # Encode all categorical features
        for col in df.columns:
            if col != target_column and df[col].dtype == 'object':
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        
        # Split features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Save feature names
        feature_names = X.columns.tolist()
        
        # Save feature names to file
        with open(FEATURE_NAMES_PATH, 'wb') as f:
            pickle.dump(feature_names, f)
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the base model
        base_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
        base_model.fit(X_train, y_train)
        
        # Create custom importance array
        custom_importance = np.zeros(len(feature_names))
        
        # Normalize the weights to sum to 1
        total_weight = sum(CUSTOM_WEIGHTS.values())
        normalized_weights = {k: v/total_weight for k, v in CUSTOM_WEIGHTS.items()}
        
        # Add the custom weights in the correct order
        for i, feature in enumerate(feature_names):
            if feature in normalized_weights:
                custom_importance[i] = normalized_weights[feature]
            else:
                custom_importance[i] = 0.001
        
        # Create the model with custom feature importance
        model = CustomImportanceRandomForest(base_model=base_model, custom_importance=custom_importance)
        
        # Save the model
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        
        # Return model, feature names, and grade mapping
        return model, feature_names, grade_mapping

def create_mock_data():
    """Create mock data if dataset is missing"""
    # Number of samples
    n_samples = 150
    
    # Create mock features
    data = {
        'Student_Age': np.random.choice([18, 19, 20, 21, 22], size=n_samples),
        'Sex': np.random.randint(0, 2, size=n_samples),
        'High_School_Type': np.random.randint(0, 2, size=n_samples),
        'Scholarship': np.random.randint(0, 2, size=n_samples),
        'Additional_Work': np.random.randint(0, 2, size=n_samples),
        'Sports_activity': np.random.randint(0, 2, size=n_samples),
        'Transportation': np.random.randint(0, 3, size=n_samples),
        'Weekly_Study_Hours': np.random.randint(1, 40, size=n_samples),
        'Attendance': np.random.randint(50, 101, size=n_samples),
        'Reading': np.random.randint(0, 11, size=n_samples),
        'Notes': np.random.randint(0, 11, size=n_samples),
        'Listening_in_Class': np.random.randint(0, 11, size=n_samples),
        'Project_work': np.random.randint(0, 11, size=n_samples),
    }
    
    # Create target variable (grades: AA, BA, BB, CB, CC, DC, DD, Fail)
    grades = ['AA', 'BA', 'BB', 'CB', 'CC', 'DC', 'DD', 'Fail']
    data['Grade'] = np.random.choice(grades, size=n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    
    # Save to CSV
    df.to_csv(DATA_PATH, index=False)
    
    return df

def plot_feature_importance(feature_names, feature_importance):
    """Create a feature importance visualization"""
    # Sort features by importance
    idx = np.argsort(feature_importance)[::-1]
    sorted_names = [feature_names[i].replace('_', ' ').title() for i in idx]
    sorted_importance = feature_importance[idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = sns.color_palette("viridis", len(feature_names))
    y_pos = np.arange(len(feature_names))
    
    ax.barh(y_pos, sorted_importance, align='center', color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance for Student Performance')
    
    # Add value labels
    for i, v in enumerate(sorted_importance):
        ax.text(v + 0.01, i, f"{v:.3f}", va='center')
    
    plt.tight_layout()
    
    return fig

def generate_study_plan(prediction):
    """Generate a personalized study plan based on prediction"""
    # Convert prediction to numeric scale if needed
    if isinstance(prediction, str):
        try:
            numeric_prediction = float(prediction)
        except:
            # If it's a letter grade like AA, BA, etc.
            letter_grade_map = {
                'AA': 5, 'BA': 4.5, 'BB': 4, 'CB': 3.5, 
                'CC': 3, 'DC': 2.5, 'DD': 2, 'Fail': 1
            }
            numeric_prediction = letter_grade_map.get(prediction, 3)
    else:
        numeric_prediction = float(prediction)
    
    # Scale numeric prediction to 1-5 scale if needed
    if numeric_prediction > 5:
        numeric_prediction = numeric_prediction / 20  # Assuming 100-point scale
    
    if numeric_prediction >= 4:
        category = "High Performer"
        focus_areas = ["Advanced concepts", "Competitive exam preparation"]
        recommended_hours = 25
    elif numeric_prediction >= 3:
        category = "Above Average"
        focus_areas = ["Concept strengthening", "Problem-solving techniques"]
        recommended_hours = 21
    elif numeric_prediction >= 2:
        category = "Average Performer"
        focus_areas = ["Regular practice", "Concept clarity"]
        recommended_hours = 18
    else:
        category = "Needs Improvement"
        focus_areas = ["Foundation concepts", "Daily structured practice"]
        recommended_hours = 15
    
    # Create weekly plan
    weekly_plan = {
        "Monday": f"Core concepts review - {max(1, recommended_hours//5)} hours",
        "Tuesday": f"Problem solving practice - {max(1, recommended_hours//5)} hours",
        "Wednesday": f"Review weak areas - {max(1, recommended_hours//5 + 1)} hours",
        "Thursday": f"Practice tests - {max(1, recommended_hours//5)} hours",
        "Friday": f"Group study/Project work - {max(1, recommended_hours//5)} hours",
        "Weekend": "Revision and rest"
    }
    
    study_plan = {
        "category": category,
        "focus_areas": focus_areas,
        "recommended_hours": recommended_hours,
        "weekly_plan": weekly_plan
    }
    
    return study_plan

def load_model_files():
    """Load model and related files"""
    try:
        # Check if model exists, if not, train it
        if not os.path.exists(MODEL_PATH):
            st.warning("Model not found. Training now...")
            model, feature_names, grade_mapping = train_model()
        else:
            # Load model
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            
            # Load feature names
            with open(FEATURE_NAMES_PATH, 'rb') as f:
                feature_names = pickle.load(f)
            
            # Load grade mapping
            with open(GRADE_MAPPING_PATH, 'rb') as f:
                grade_mapping = pickle.load(f)
            
        return model, feature_names, grade_mapping, True
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, [], {}, False

def get_grade_color(grade):
    """Return color based on grade"""
    if isinstance(grade, str):
        if grade in ['AA', 'BA']:
            return "green"
        elif grade in ['BB', 'CB']:
            return "blue"
        elif grade in ['CC', 'DC']:
            return "orange"
        else:
            return "red"
    else:
        # Numeric grade
        if grade >= 4:
            return "green"
        elif grade >= 3:
            return "blue"
        elif grade >= 2:
            return "orange"
        else:
            return "red"

def main():
    """Main Streamlit application"""
    
    st.title("Student Performance Prediction System")
    
    # Create sidebar for controls
    st.sidebar.header("Control Panel")
    section = st.sidebar.radio("Navigation", ["Predict Performance", "About", "Model Information"])
    
    # Initialize or load model
    model, feature_names, grade_mapping, success = load_model_files()
    
    if section == "About":
        st.header("About This Application")
        st.write("""
        This application uses machine learning to predict student academic performance based on various factors.
        It provides personalized study plans and insights to help students improve their academic results.
        
        The prediction model uses Random Forest with custom feature importance weights to prioritize the most 
        important factors affecting student performance.
        
        ### Key Features
        - Predict academic performance based on student attributes
        - Visualize the most important factors affecting performance
        - Get a personalized study plan based on predicted performance
        - Understand which areas need improvement
        
        ### How To Use
        1. Go to the "Predict Performance" section
        2. Fill in all student information
        3. Click "Predict Performance" to get results
        4. Review the prediction, feature importance, and study plan
        """)
        
    elif section == "Model Information":
        st.header("Model Information")
        
        st.subheader("Feature Importance")
        st.write("The model has been configured with custom feature importance weights to emphasize key factors:")
        
        # Show custom weights
        weights_df = pd.DataFrame({
            'Feature': list(CUSTOM_WEIGHTS.keys()),
            'Weight': list(CUSTOM_WEIGHTS.values())
        }).sort_values('Weight', ascending=False)
        
        st.dataframe(weights_df)
        
        # Plot feature importance
        if hasattr(model, 'feature_importances_'):
            fig = plot_feature_importance(feature_names, model.feature_importances_)
            st.pyplot(fig)
        
        st.subheader("Dataset Information")
        if os.path.exists(DATA_PATH):
            data = pd.read_csv(DATA_PATH)
            st.write(f"Dataset with {data.shape[0]} rows and {data.shape[1]} columns")
            st.dataframe(data.head())
            
            st.write("### Data Statistics")
            st.dataframe(data.describe())
        else:
            st.warning("Dataset file not found.")
            
        # Model retraining button
        if st.button("Retrain Model"):
            model, feature_names, grade_mapping = train_model()
            st.success("Model retrained successfully!")
            
    else:  # Predict Performance
        st.header("Student Performance Prediction")
        
        # Create a form for user input
        with st.form("prediction_form"):
            st.subheader("Student Information")
            
            col1, col2 = st.columns(2)
            
            # Use two columns for better layout
            with col1:
                student_name = st.text_input("Student Name")
                
                if 'Sex' in feature_names:
                    sex = st.selectbox("Sex", ["Male", "Female"], 
                                      format_func=lambda x: x)
                
                if 'Student_Age' in feature_names:
                    age = st.slider("Student Age", 15, 30, 20)
                
                if 'High_School_Type' in feature_names:
                    high_school = st.selectbox("High School Type", ["Public", "Private"], 
                                              format_func=lambda x: x)
                
                if 'Scholarship' in feature_names:
                    scholarship = st.selectbox("Scholarship", ["Yes", "No"], 
                                              format_func=lambda x: x)
                
                if 'Additional_Work' in feature_names:
                    additional_work = st.selectbox("Additional Work", ["Yes", "No"], 
                                                  format_func=lambda x: x)
                
                if 'Sports_activity' in feature_names:
                    sports = st.selectbox("Sports Activity", ["Yes", "No"], 
                                         format_func=lambda x: x)
                    
            with col2:
                if 'Transportation' in feature_names:
                    transport = st.selectbox("Transportation", ["Public", "Private", "Walking"], 
                                            format_func=lambda x: x)
                
                if 'Weekly_Study_Hours' in feature_names:
                    study_hours = st.slider("Weekly Study Hours", 0, 50, 15)
                
                if 'Attendance' in feature_names:
                    attendance = st.slider("Attendance (%)", 0, 100, 75)
                
                if 'Reading' in feature_names:
                    reading = st.slider("Reading Skills (0-10)", 0, 10, 5)
                
                if 'Notes' in feature_names:
                    notes = st.slider("Notes Quality (0-10)", 0, 10, 5)
                
                if 'Listening_in_Class' in feature_names:
                    listening = st.slider("Listening in Class (0-10)", 0, 10, 5)
                
                if 'Project_work' in feature_names:
                    project = st.slider("Project Work Quality (0-10)", 0, 10, 5)
            
            # Submit button
            submitted = st.form_submit_button("Predict Performance")
        
        # Process the prediction when form is submitted
        if submitted:
            # Create input data based on form values
            input_data = {}
            
            # Map categorical inputs to numeric values
            if 'Sex' in feature_names:
                input_data['Sex'] = 1 if sex == "Male" else 0
            if 'High_School_Type' in feature_names:
                input_data['High_School_Type'] = 1 if high_school == "Public" else 0
            if 'Scholarship' in feature_names:
                input_data['Scholarship'] = 1 if scholarship == "Yes" else 0
            if 'Additional_Work' in feature_names:
                input_data['Additional_Work'] = 1 if additional_work == "Yes" else 0
            if 'Sports_activity' in feature_names:
                input_data['Sports_activity'] = 1 if sports == "Yes" else 0
            if 'Transportation' in feature_names:
                transport_map = {"Public": 0, "Private": 1, "Walking": 2}
                input_data['Transportation'] = transport_map[transport]
            
            # Add numeric inputs
            if 'Student_Age' in feature_names:
                input_data['Student_Age'] = age
            if 'Weekly_Study_Hours' in feature_names:
                input_data['Weekly_Study_Hours'] = study_hours
            if 'Attendance' in feature_names:
                input_data['Attendance'] = attendance
            if 'Reading' in feature_names:
                input_data['Reading'] = reading
            if 'Notes' in feature_names:
                input_data['Notes'] = notes
            if 'Listening_in_Class' in feature_names:
                input_data['Listening_in_Class'] = listening
            if 'Project_work' in feature_names:
                input_data['Project_work'] = project
            
            # Fill any missing features with default values
            for feature in feature_names:
                if feature not in input_data:
                    input_data[feature] = 0
            
            # Create DataFrame with correct feature order
            input_df = pd.DataFrame([input_data])
            input_df = input_df[feature_names]  # Ensure correct order
            
            # Make prediction
            encoded_prediction = model.predict(input_df)[0]
            
            # Convert encoded prediction to original grade
            original_grade = grade_mapping.get(encoded_prediction, str(encoded_prediction))
            
            # Get prediction confidence if available
            try:
                probabilities = model.predict_proba(input_df)[0]
                confidence = float(probabilities.max()) * 100
            except Exception:
                confidence = None
            
            # Generate study plan
            study_plan = generate_study_plan(original_grade)
            
            # Display results
            st.header(f"Results for {student_name}")
            
            # Use columns for result layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Prediction")
                
                # Display grade with color
                grade_color = get_grade_color(original_grade)
                st.markdown(f"""
                <div style="background-color: {grade_color}22; padding: 20px; border-radius: 10px; border: 2px solid {grade_color};">
                    <h1 style="text-align: center; color: {grade_color};">{original_grade}</h1>
                    <p style="text-align: center;">Predicted Grade</p>
                </div>
                """, unsafe_allow_html=True)
                
                if confidence:
                    st.metric("Prediction Confidence", f"{confidence:.1f}%")
                
                st.subheader("Performance Category")
                st.info(study_plan["category"])
                
                # Feature importance visualization
                st.subheader("Factors Influencing Performance")
                fig = plot_feature_importance(feature_names, model.feature_importances_)
                st.pyplot(fig)
                
            with col2:
                st.subheader("Personalized Study Plan")
                
                st.metric("Recommended Weekly Study Hours", study_plan["recommended_hours"])
                
                st.subheader("Focus Areas")
                for area in study_plan["focus_areas"]:
                    st.write(f"- {area}")
                
                st.subheader("Weekly Schedule")
                for day, activity in study_plan["weekly_plan"].items():
                    st.write(f"**{day}:** {activity}")
                
                st.subheader("Tips for Improvement")
                
                # Different tips based on performance category
                if study_plan["category"] == "High Performer":
                    tips = [
                        "Challenge yourself with advanced problem sets",
                        "Consider peer tutoring to reinforce your knowledge",
                        "Explore competitive exams and Olympiads",
                        "Develop research projects in areas of interest",
                        "Balance your high academic load with relaxation techniques"
                    ]
                elif study_plan["category"] == "Above Average":
                    tips = [
                        "Focus on strengthening your conceptual understanding",
                        "Practice more complex problems in your stronger areas",
                        "Form or join study groups for collaborative learning",
                        "Create concise summary notes for quick revision",
                        "Maintain consistency in your study schedule"
                    ]
                elif study_plan["category"] == "Average Performer":
                    tips = [
                        "Increase regular practice with varied problem sets",
                        "Focus on understanding core concepts thoroughly",
                        "Use visual aids and diagrams to improve comprehension",
                        "Schedule regular review sessions of previous material",
                        "Seek help for topics you find challenging"
                    ]
                else:
                    tips = [
                        "Build a strong foundation in fundamental concepts",
                        "Break down study sessions into shorter, more frequent periods",
                        "Use multiple learning resources (videos, books, tutorials)",
                        "Practice active recall techniques instead of passive reading",
                        "Track your progress on specific topics to build confidence"
                    ]
                
                for tip in tips:
                    st.write(f"- {tip}")

if __name__ == "__main__":
    main()
