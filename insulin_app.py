import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(layout='wide', page_title='Insulin Dosage Predictor App', page_icon="IDPA logo.svg")

## data loading
@st.cache_data
def load_data():
    URL = './insulin_dosage_prediction.csv'
    df = pd.read_csv(URL)
    return df

df = load_data()

model = joblib.load('./insulin_dosage_predictor.joblib')

@st.cache_resource
def train_model(data):
    num_cols = ['age', 'glucose_level', 'physical_activity', 'BMI', 'HbA1c', 'weight', 'insulin_sensitivity', 'sleep_hours', 'creatinine']
    nominal_cols = ['gender', 'previous_medications']
    ordinal_cols = ['family_history', 'food_intake']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('nom', OneHotEncoder(sparse_output=False), nominal_cols),
            ('ord', OrdinalEncoder(), ordinal_cols)
        ]
    )
    
    x = data.drop('Insulin', axis=1)
    y = data['Insulin']
    
    x_transformed = preprocessor.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.3)
    
    model.fit(x_train, y_train)
    acc = accuracy_score(y_test, model.predict(x_test))
    return acc
    
accuracy = train_model(df)

st.sidebar.title('Insulin Dosage Predictor App')
page = st.sidebar.radio('Navigate', ['Predictor', 'Data'])
st.sidebar.markdown('---')
st.sidebar.caption('Website created with Streamlit')

if page == 'Predictor':
    st.title("Insulin Dosage Predictor")
    st.success(f'Model Accuracy: {(accuracy*100):.4f}%')
elif page == 'Data':
    st.title("A Glance At The Dataset")
    st.markdown('Explore the dataset.')
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(['Overview', 'Distributions', 'Raw Data'])
    
    with tab1: 
        fig1 = px.histogram(df, x='BMI', y='glucose_level', color='Insulin', hover_data=['BMI', 'glucose_level'], title='BMI vs Glucose')
        st.plotly_chart(fig1, use_container_width=True)

        c1_1, c1_2 = st.columns([2/5, 3/5])
        with c1_1: 
            fig2 = px.histogram(df, x='family_history', y='glucose_level', color='Insulin', hover_data=['family_history', 'glucose_level'], title='Genetics vs Glucose')
            st.plotly_chart(fig2, use_container_width=True)
        with c1_2: 
            fig3 = px.histogram(df, x='food_intake', y='glucose_level', color='Insulin', hover_data=['food_intake', 'glucose_level'], title='Food Intake vs Glucose')
            st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.histogram(df, x='insulin_sensitivity', y='glucose_level', color='Insulin', hover_data=['insulin_sensitivity', 'glucose_level'], title='Insulin Sensitivity vs Glucose')
        st.plotly_chart(fig4, use_container_width=True)

        fig5 = px.histogram(df, x='weight', y='glucose_level', color='Insulin', hover_data=['weight', 'glucose_level'], title='Weight vs Glucose')
        st.plotly_chart(fig5, use_container_width=True)










