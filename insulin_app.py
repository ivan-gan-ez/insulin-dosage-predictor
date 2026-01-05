import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

st.set_page_config(layout='wide', page_title='Insulin Dosage Predictor App', page_icon="IDPA logo.svg")

## data loading
@st.cache_data
def load_data():
    URL = './insulin_dosage_prediction.csv'
    df = pd.read_csv(URL)
    return df

df = load_data().drop('patient_id', axis=1)

model = joblib.load('./insulin_dosage_predictor_v2.joblib')

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

@st.cache_resource
def train_model(data):
    x = data.drop('Insulin', axis=1)
    y = data['Insulin']
    
    x_transformed = preprocessor.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.3)
    model.fit(x_train, y_train)
    acc = accuracy_score(y_test, model.predict(x_test))
    return model, acc
    
model, accuracy = train_model(df)

st.logo('IDPA logo.svg')
st.sidebar.title('Insulin Dosage Predictor App')
page = st.sidebar.radio('Navigate', ['Predictor', 'Data'])
st.sidebar.markdown('---')
st.sidebar.caption('Website created with Streamlit')

if page == 'Predictor':
    st.title("Insulin Dosage Predictor")
    st.success(f'Model Accuracy: {(accuracy*100):.4f}%')

    with st.form('prediction_form'):
        c1, c2, c3 = st.columns(3, vertical_alignment="center")
        gender = c1.selectbox('Gender', ['male', 'female'])
        age = c2.slider('Age (years)', min_value=18, max_value=80, value=35, step=1)
        family_history = c3.selectbox('Family has history of diabetes?', ['yes', 'no'], index=1)
        
        c4, c5, c6 = st.columns(3)
        glucose_level = c4.number_input('Glucose level (mg/dl)', min_value=30.0, max_value=300.0, value=140.0, step=0.01)
        physical_activity = c5.slider('Physical activity (hours)', min_value=0.0, max_value=12.0, value=2.0, step=0.05)
        food_intake = c6.selectbox('Food intake', ['low', 'medium', 'high'])
        
        c7, c8, c9 = st.columns(3)
        previous_medications = c7.selectbox('Previous medications', ['none', 'oral', 'insulin', 'both'])
        BMI = c8.slider('BMI (kg/m²)', min_value=14.0, max_value=42.0, value=30.0, step=0.01)
        HbA1c = c9.slider('Haemoglobin A1c level (%)', min_value=0.0, max_value=20.0, value=8.0, step=0.01)

        c10, c11 = st.columns(2)
        weight = c10.number_input('Weight (kg)', min_value=30.0, max_value=150.0, value=80.0, step=0.01)
        insulin_sensitivity = c11.slider('Insulin sensitivity', min_value=0.0, max_value=5.0, value=1.5, step=0.01)

        c12, c13 = st.columns(2)
        sleep_hours = c12.slider('Hours of sleep per day', min_value=0.0, max_value=12.0, value=8.0, step=0.25)
        creatinine = c13.slider('Serum creatinine level (mg/dL)', min_value=1.0, max_value=5.0, value=1.3, step=0.01)
        
        submit_btn = st.form_submit_button('Analyse Risk', type='primary')

    st.divider()

    if submit_btn:
        input_data = pd.DataFrame([[gender, age, family_history, glucose_level, physical_activity, food_intake, previous_medications, BMI, HbA1c, weight, insulin_sensitivity, sleep_hours, creatinine]], columns=df.columns[:-1])
        preprocessor.fit(df.drop('Insulin', axis=1))
        transformed_data = preprocessor.transform(input_data)
        prediction = model.predict(transformed_data)[0]
        if prediction == 'steady':
            st.warning(f"You require a steady insulin dosage.")
        elif prediction == 'up':
            st.error(f"You require a higher insulin dosage.")
        elif prediction == 'down':
            st.info(f"You require a lower insulin dosage.")
        else:
            st.success(f"You require {prediction} insulin dosage.")

elif page == 'Data':
    st.title("A Glance At The Dataset")
    st.markdown('Explore the dataset.')
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(['Overview', 'Distributions', 'Raw Data'])
    
    with tab1: 
        c1, c2, c3, c4, c5, c6 = st.columns([4/28, 5/28, 5/28, 4/28, 6/28, 4/28])
        with c1:
            st.metric('Total Records', len(df),'Data size', delta_color='off', delta_arrow="off")
        with c2:
            st.metric('Avg. Glucose Level', f"{df['glucose_level'].mean():.1f}",'mg/dl', delta_color='off', delta_arrow="off")
        with c3:
            st.metric('Avg. Physical Activity', f"{df['physical_activity'].mean():.2f}",'hours', delta_color='off', delta_arrow="off")
        with c4:
            st.metric('Avg. BMI', f"{df['BMI'].mean():.2f}",'kg/m²', delta_color='off', delta_arrow="off")
        with c5:
            st.metric('Avg. Haemoglobin A1c Level', f"{df['HbA1c'].mean():.2f}",'%', delta_color='off', delta_arrow="off")
        with c6:
            st.metric('Avg. Weight', f"{df['weight'].mean():.2f}",'kg', delta_color='off', delta_arrow="off")
        
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

    with tab2: 
        st.markdown("Distribution of various variables in the dataset")
        c2_1, c2_2 = st.columns(2)
        
        with c2_1: 
            for col in df.columns[::2]:
                fig = px.histogram(df, x=col, color="Insulin", title=col.replace("_", " ").capitalize())
                st.plotly_chart(fig, use_container_width=True)

        with c2_2: 
            for col in df.columns[1:-1:2]:
                fig = px.histogram(df, x=col, color="Insulin", title=col.replace("_", " ").capitalize())
                st.plotly_chart(fig, use_container_width=True)
            insulin_fig = px.histogram(df, x='Insulin', title='Insulin')
            st.plotly_chart(insulin_fig, use_container_width=True)

    with tab3: 
            st.subheader('Raw data')
            st.markdown('A sample of 50 rows of data from the dataset.')
            st.dataframe(df.sample(50))
            st.markdown('Source: https://www.kaggle.com/datasets/robyburns/insulin-dosage')









