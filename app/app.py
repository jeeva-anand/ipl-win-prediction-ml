import pickle as pkl
import streamlit as st
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']



cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']


pipe = pkl.load(open('./trained_model.pkl','rb'))

st.title('Real-Time IPL Match Outcome Prediction')

col1,col2 = st.columns(2)


with col1:
  battingteam = st.selectbox('Select the bating team',sorted(teams))

with col2:
  bowlingteam = st.selectbox('Select the bowling team',sorted(teams))


city = st.selectbox('Select the city',sorted(cities))

target = int(st.number_input('Target',step=1))


col1, col2, col3 = st.columns(3)

with col1:
  score = int(st.number_input('score',step=1))

with col2:
  overs = int(st.number_input('Over completed',step=1))

with col3:
  wickets = int(st.number_input('wicket fallen',step=1))


if score > target:
    st.write(battingteam,"won the match")

elif score == target-1 and overs == 20:
    st.write("Match Drawn")

elif wickets == 10 and score < target-1:
    st.write(bowlingteam, 'Won the match')

elif wickets == 10 and score == target-1:
    st.write('Match tied')

elif battingteam==bowlingteam:
    st.write('To proceed, please select different teams because no match can be played between the same teams')

else:

    # Checking if the input values are valid or not
    if target >= 0 and target <= 300  and overs >= 0 and overs <=20 and wickets <= 10 and wickets>=0 and score>= 0:
      try:
        if st.button('Predict probability'):
          runs_left = target-score
          # Calculating the number of balls left
          balls_left = 120-(overs*6)

                # Calculating the number of wickets left for the batting team
          wickets = 10-wickets

                # Calculating the current Run-Rate of the batting team
          currentrunrate = score/overs

                # Calculating the Required Run-Rate for the batting team to win
          requiredrunrate = (runs_left*6)/balls_left

                # Creating a pandas DataFrame containing the user inputs
          input_df = pd.DataFrame(
                               {'batting_team': [battingteam],
                                'bowling_team': [bowlingteam],
                                'city': [city],
                                'runs_left': [runs_left],
                                'balls_left': [balls_left],
                                'wickets': [wickets],
                                'total_runs_x': [target],
                                'cur_run_rate': [currentrunrate],
                                'req_run_rate': [requiredrunrate]})
                # Loading the trained machine learning pipeline to make the prediction
          result = pipe.predict_proba(input_df)

                # Extracting the likelihood of loss and win
          lossprob = result[0][0]
          winprob = result[0][1]
          st.header(battingteam+"- "+str(round(winprob*100))+"%")

          st.header(bowlingteam+"- "+str(round(lossprob*100))+"%")


        #Catching ZeroDivisionError
      except ZeroDivisionError:
            st.error("Please fill all the details")

    #Displaying an error message if the input is incorrect
      else:
        st.error('There is something wrong with the input, please fill the correct details')


