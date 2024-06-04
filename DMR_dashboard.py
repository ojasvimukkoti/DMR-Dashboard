import os
import webbrowser
import time
import shutil
import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
from io import BytesIO
import urllib.request
from DMR_presentation import DMRAnalysis


#creating initial set-up of streamlit app
st.set_page_config(layout='wide',
                   initial_sidebar_state="expanded")
st.info("This is a sample of what the dashboard looks like. The official one requires users to upload\
    the DMR Log.")
st.title("Analysis of DMR Log")


@st.cache_data
def create_bar_chart(df, x_col, y_col, title, xaxis_title, yaxis_title, log_scale = True):
    bar_chart=px.bar(
                    df,
                    x=df.index,
                    y=y_col,
                    title = title,
                    labels={y_col:yaxis_title},
                    text=y_col
    )
    if log_scale:
        bar_chart.update_layout(
            xaxis=dict(tickmode='array',tickvals=df[x_col].tolist()),
            xaxis_title=xaxis_title,
            yaxis_title = yaxis_title,
            yaxis_type = 'log',
            bargap=0.1
        )
    else:
        bar_chart.update_layout(
            xaxis=dict(tickmode='linear'),
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title
        )

    bar_chart.update_traces(textposition='outside', texttemplate='%{y}',
                            textfont=dict(color='black'))
    
    bar_chart.update_layout(
        xaxis_tickvals=df.index.tolist(),
        xaxis_ticktext=df['DMR #'].tolist()
    )

    return bar_chart

#Function that will check if a date is in a specfied valid format
def valid_date(date_string,date_format):
    try:
        pd.to_datetime(date_string, format=date_format)
        return True
    except ValueError:
        return False

@st.cache_data
def generate_and_downlaod_presentation(data_df):
    #creating object from class
    dmr_analysis = DMRAnalysis(data_df)

    return dmr_analysis.generate_presentation()

filtered_data = pd.read_excel("Example_DMR_Log.xlsx")

def main():
    #columns for the histograms
    col = st.columns([1, 1])
    colm = st.columns([1,1])
    if filtered_data is not None:
        #changing the data types of columns to different types that will work
        filtered_data['Category'] = filtered_data['Category'].astype('category')
        filtered_data["Disposition"] = filtered_data["Disposition"].astype("category")
        filtered_data['Status'] = filtered_data["Status"].astype('category')
        filtered_data["Originator"] = filtered_data["Originator"].astype('str')
        filtered_data["Owner"] = filtered_data["Owner"].astype('str')

        #First Graph - histogram showing the age of all OPEN DMRs that do not have disposition

        #Defined the ecpected date format
        expected_date_format = '%m/%d/%Y'

        #Creating list that will hold the invalid date information
        invalid_date_info = []

        #For-loop that will iterate through the Date column and see if the dates are in the correct format
        for index, date_value in filtered_data['Date'].items():
            if not valid_date(date_value, expected_date_format):
                #stores the index of invalid date and the value 
                invalid_date_info.append(date_value)
        #filtering out the dates that not in the correct format and converting the remaining dates
        filtered_data['Date'] = filtered_data['Date'].apply(lambda x: x if valid_date(x, expected_date_format) else pd.NaT)
        #dropping any rows in the data that have 'NaT' as its values in the Date column
        filtered_data.dropna(subset=['Date'], inplace=True)


        #calculating the Age(current date - date) of all DMRs
        current_date = datetime.now()
        formatted_current_date = current_date.strftime('%m-%d-%Y')
        #adding a new column in the filtered_data DF that stores the Age in years
        filtered_data['Age of DMRs (Years)'] = (pd.to_datetime(formatted_current_date) - pd.to_datetime(filtered_data['Date'])).dt.days / 365.25
        filtered_data['Age of DMRs (Years)'] = round(filtered_data['Age of DMRs (Years)'], 1)
        
        #adding a new column in the filteredt DF that has the age of the DMRs in DAYS
        filtered_data['Age of DMRs (Days)'] = (pd.to_datetime(formatted_current_date)-pd.to_datetime(filtered_data['Date'])).dt.days
        filtered_data['Age of DMRs (Days)'] = round(filtered_data['Age of DMRs (Days)'], 1)

        #doing the necessary value replacement for the Status column
        replace_status_map = {'Closed':'CLOSED', 'Open': 'OPEN', 'OPEN ': 'OPEN'}
        filtered_data['Status'] = filtered_data['Status'].replace(replace_status_map).str.upper().fillna('OTHER')
        #getting the needed data for the Histogram
        open_DMRs_no_disposition = filtered_data[(filtered_data['Disposition'].isnull()) & (filtered_data['Status'] == 'OPEN')]
        open_DMRs_w_disposition = filtered_data[~(filtered_data['Disposition'].isnull())& (filtered_data['Status']=='OPEN')]

        # Calculate the number of bins dynamically (square root of the number of data points)
        num_bins = int(np.sqrt(len(filtered_data)))

        #calculating the mean
        mean_no_disposition = open_DMRs_no_disposition['Age of DMRs (Years)'].mean()
        mean_w_disposition = open_DMRs_w_disposition['Age of DMRs (Years)'].mean()
        
        display_age_box_histogram = st.checkbox('Click to see histogram in Days')
        #defining a variable that will change the age defintion w/ an If-statement
        age_col = 'Age of DMRs (Days)' if display_age_box_histogram else 'Age of DMRs (Years)'

        # with colm[0]:
            #creating histogram of the OPEN DMRs 
        histogram_DMRs = px.histogram(
                open_DMRs_no_disposition, x=age_col, color='Status',
                nbins=num_bins,
                labels = {'Age of DMRs':'Age of DMRs(Years)', 'Count':'Frequency'},
                title='Histogram of OPEN DMRs Ages with and without Disposition',
                color_discrete_map={'OPEN': 'blue'},
                barmode="overlay"
            )
            #adding a 2nd histogram for the DMRs w/ disposition
        histogram_DMRs.add_trace(
                px.histogram(
                    open_DMRs_w_disposition, x=age_col, color = 'Status',
                    nbins=num_bins, color_discrete_map={'OPEN': 'red'},
                    barmode="overlay"
                ).data[0]
            )
            #adjusting the legend
        histogram_DMRs.update_layout(
                legend=dict(x=0, y=-1)
            )
            #adding legend for the DMRs w/ and w/o Dispositions
        histogram_DMRs.for_each_trace(lambda t: t.update(legendgroup=t.name))

        histogram_DMRs.data[0].update(name='DMRs with no Disposition')
        histogram_DMRs.data[1].update(name='DMRs with Disposition')
            
            #Adding vertical line for mean of DMRs with no disposition
        histogram_DMRs.add_shape(
                go.layout.Shape(
                    type='line',
                    x0=mean_no_disposition,
                    x1=mean_no_disposition,
                    y0=0,
                    y1=1,
                    xref='x',
                    yref='paper',
                    line=dict(color='blue', width=2)
                )
            )
            #Adding vertical line for mean of DMRs with disposition
        histogram_DMRs.add_shape(
                go.layout.Shape(
                    type='line',
                    x0=mean_w_disposition,
                    x1=mean_w_disposition,
                    y0=0,
                    y1=1,
                    xref='x',
                    yref='paper',
                    line=dict(color='red', width=2)
                )
            )
            #adding arrows that point to respective mean on figure
        histogram_DMRs.add_annotation(
                x=mean_no_disposition, y =1.02,
                text=f'Mean: {round(mean_no_disposition,3)}',
                showarrow = True,
                xref='x',
                yref='paper',
                font= dict(color='blue')
            )
        histogram_DMRs.add_annotation(
                x=mean_w_disposition, y=1.02,
                text = f'Mean: {round(mean_w_disposition, 3)}',
                showarrow=True,
                xref='x',
                yref = 'paper',
                font=dict(color='red')
            )
        st.plotly_chart(histogram_DMRs, use_container_width=True) #displaying the plot

        #2nd graph - bar chart of counts of DMRs w/ relation
        #to each of the 6 dispositions

        replace_map = {'Rework':'REWORK', 'rework': 'REWORK', 'scrap': 'SCRAP',
                    'REWORk':'REWORK', 'SCRAP ':'SCRAP', "":'OTHER'}

        filtered_data['Disposition'] = filtered_data['Disposition'].replace(replace_map).str.upper().fillna('OTHER')

        filtered_data_disposition = filtered_data[~((filtered_data['Disposition']=='CANCELLED') | (filtered_data['Disposition']=='OTHER'))]

        my_expan = st.expander(label="Click to see the DMRs Counts Bar Chart")

        cols_expander = st.columns([1,1])
        cols_expander1 = st.columns([1,1])

        # with cols_expander[0].expander(label="Click to see the DMRs Counts Bar Chart"):
        with my_expan:
                cols2 = st.columns([2,1], gap = 'large')
                countS_disposition = filtered_data_disposition['Disposition'].value_counts().reset_index()
                countS_disposition.columns = ['Disposition', 'Count']

                # Sort the DataFrame in descending order based on the 'Count' column
                countS_disposition = countS_disposition.sort_values(by='Count', ascending=False)


                disposition_bar_chart = px.bar(
                                countS_disposition,
                                x = 'Disposition',
                                y='Count',
                                labels={'Disposition':'Disposition', 'index':'Count'},
                                title = 'DMR Counts by Disposition',
                                color_discrete_sequence=['dodgerblue']
                                )                                    

                disposition_bar_chart.update_traces(text=countS_disposition['Count'], textposition='outside')
                
                cols2[0].plotly_chart(disposition_bar_chart, use_container_width=True)

                if cols2[0].checkbox('Show Counts for DMRs Disposition'):
                    cols2[0].write(countS_disposition)


        #ordering the data to descending order
        descending_ordered_data = filtered_data.sort_values(by='Age of DMRs (Years)', ascending=False)
        #getting the top 10 DMRs
        oldest_DMRs_closed = descending_ordered_data[descending_ordered_data['Status']=='CLOSED']
        oldest_closed_DMR = oldest_DMRs_closed.head(10).reset_index()
        #arranging the dataframe of the top 10
        oldest_CLOSEDDMRs = oldest_closed_DMR[['DMR #', 'Part #', 'Originator', 'Disposition', 'Status', 'Age of DMRs (Years)']]

        #getting the data for the oldest DMRs that have Status of "OPEN"
        oldest_DMRs_open = descending_ordered_data[descending_ordered_data['Status']=='OPEN']
        oldest_open_DMR = oldest_DMRs_open.head(10).reset_index()
        oldest_OPENDMRs = oldest_open_DMR[['DMR #', 'Part #', 'Originator', 'Disposition', 'Status', 'Age of DMRs (Years)']]

        #adding the age of dmrs in days to the open/closed dmrs dataframe
        oldest_OPENDMRs['Age of DMRs (Days)'] = filtered_data['Age of DMRs (Days)']
        oldest_CLOSEDDMRs['Age of DMRs (Days)'] = filtered_data['Age of DMRs (Days)']
        
        #Need top 10 oldest DMRs by age
            #DF: DMR Num, Quality Engineer, disposition status

        my_expanderr = st.expander(label='Click to see the Top 10 oldest DMRs')
        with my_expanderr:
            st.subheader("Top 10 Oldest DMRs")
            col = st.columns([1,1], gap = 'small')
            with col[0]:
                select_data = st.radio("Select which Top 10 Oldest DMRs Data to Display",
                                    ['Top 10 Oldest **OPEN** DMRs', 'Top 10 Oldest **CLOSED** DMRs'])
                with col[1]:
                    #checkbox that will change the graph to display age in years to days
                    display_age_box = st.checkbox('Click to see the chart in Days')

            if select_data == "Top 10 Oldest **OPEN** DMRs":

                cols = st.columns([1,1], gap = 'small')

                #defining a variable that will change the age defintion w/ an If-statement
                age_col = 'Age of DMRs (Days)' if display_age_box else 'Age of DMRs (Years)'

                top10_OPENbar_chart_age = create_bar_chart(
                        oldest_OPENDMRs,
                        x_col='DMR #',
                        y_col=age_col,
                        title='Top 10 Oldest OPEN DMRs by Age Chart',
                        xaxis_title='DMR Number',
                        yaxis_title=age_col,
                        log_scale=True
                    )
                with cols[0]:
                        st.plotly_chart(top10_OPENbar_chart_age, use_container_width=True)
                with cols[1]:
                        st.write("Top 10 Oldest **OPEN** DMRs Data")
                        st.write(oldest_OPENDMRs)

            elif select_data == "Top 10 Oldest **CLOSED** DMRs":
                    cols1 = st.columns([1,1], gap = 'small')

                    age_col = 'Age of DMRs (Days)' if display_age_box else 'Age of DMRs (Years)'

                    top10_CLOSEDbar_chart_age= create_bar_chart(
                        oldest_CLOSEDDMRs,
                        x_col='DMR #',
                        y_col=age_col,
                        title='Top 10 Oldest CLOSED DMRs by Age Chart',
                        xaxis_title='DMR Number',
                        yaxis_title=age_col,
                        log_scale=True
                    )
                    with cols1[0]:
                        st.plotly_chart(top10_CLOSEDbar_chart_age, use_container_width=True)
                    with cols1[1]:
                        st.write("Top 10 Oldest **CLOSED** DMRs Data")
                        st.write(oldest_CLOSEDDMRs)

        my_ex= st.expander(label='Click to see the DMR Disposition Age Descriptive Statistics')
        with my_ex:
            cols_stats = st.columns([1,1])
            with cols_stats[0]:
                st.write("**DMR Disposition Age Descriptive Statistics in Years**")
                stats_no_disposition_yrs = {
                'Mean': open_DMRs_no_disposition['Age of DMRs (Years)'].mean(),
                'Median': open_DMRs_no_disposition['Age of DMRs (Years)'].median(),
                'Std Dev': open_DMRs_no_disposition['Age of DMRs (Years)'].std(),
                'Range': open_DMRs_no_disposition['Age of DMRs (Years)'].max() - open_DMRs_no_disposition['Age of DMRs (Years)'].min(),
                'Max (Oldest DMR)': open_DMRs_no_disposition['Age of DMRs (Years)'].max()
            }
                # Calculate statistics for DMRs with disposition
                stats_w_disposition_yrs = {
                    'Mean': open_DMRs_w_disposition['Age of DMRs (Years)'].mean(),
                    'Median': open_DMRs_w_disposition['Age of DMRs (Years)'].median(),
                    'Std Dev': open_DMRs_w_disposition['Age of DMRs (Years)'].std(),
                    'Range': open_DMRs_w_disposition['Age of DMRs (Years)'].max() - open_DMRs_w_disposition['Age of DMRs (Years)'].min(),
                    'Max (Oldest DMR)': open_DMRs_w_disposition['Age of DMRs (Years)'].max()
                }

                df_stats_no_disposition_yrs = pd.DataFrame(stats_no_disposition_yrs, index=['DMRs with no Disposition'])
                df_stats_w_disposition_yrs = pd.DataFrame(stats_w_disposition_yrs, index=['DMRs with Disposition'])

                df_combinded_stats_yrs = pd.concat([df_stats_no_disposition_yrs, df_stats_w_disposition_yrs])
                st.write(round(df_combinded_stats_yrs,2))
            with cols_stats[1]:
                st.write("**DMR Disposition Age Descriptive Statistics in Days**")
                stats_no_disposition_days = {
                'Mean': open_DMRs_no_disposition['Age of DMRs (Days)'].mean(),
                'Median': open_DMRs_no_disposition['Age of DMRs (Days)'].median(),
                'Std Dev': open_DMRs_no_disposition['Age of DMRs (Days)'].std(),
                'Range': open_DMRs_no_disposition['Age of DMRs (Days)'].max() - open_DMRs_no_disposition['Age of DMRs (Days)'].min(),
                'Max (Oldest DMR)': open_DMRs_no_disposition['Age of DMRs (Days)'].max()
            }
                # Calculate statistics for DMRs with disposition
                stats_w_disposition_days = {
                    'Mean': open_DMRs_w_disposition['Age of DMRs (Days)'].mean(),
                    'Median': open_DMRs_w_disposition['Age of DMRs (Days)'].median(),
                    'Std Dev': open_DMRs_w_disposition['Age of DMRs (Days)'].std(),
                    'Range': open_DMRs_w_disposition['Age of DMRs (Days)'].max() - open_DMRs_w_disposition['Age of DMRs (Days)'].min(),
                    'Max (Oldest DMR)': open_DMRs_w_disposition['Age of DMRs (Days)'].max()
                }

                df_stats_no_disposition_days = pd.DataFrame(stats_no_disposition_days, index=['DMRs with no Disposition'])
                df_stats_w_disposition_days = pd.DataFrame(stats_w_disposition_days, index=['DMRs with Disposition'])

                df_combinded_stats_days = pd.concat([df_stats_no_disposition_days, df_stats_w_disposition_days])
                st.write(round(df_combinded_stats_days,2))    
                       

        my_expander = st.expander(label='Expand to see the Raw Data')
        with my_expander:
            'Raw Data'
            st.write(filtered_data)
            if invalid_date_info:
                st.write(f"**Removed from Data:**\
                        \n Invalid Date Format of: {invalid_date_info}")

        #need to now add different optins for downloading open/closed dmr data
        with st.sidebar.expander("Download Options",expanded=True):
            st.subheader("Download Options")
            download_option = st.selectbox("Select Download Option",["Raw Data", "Top 10 Oldest CLOSED DMR Data","Top 10 Oldest OPEN DMR Data",
                                                                    "DMR Descriptive Statistics in Years","DMR Descriptive Statistics in Days", "Histogram of DMRs Ages Chart", 
                                                                    "DMR Counts Chart"])

            if download_option == "Raw Data" and filtered_data is not None:
                raw_csv_data = filtered_data.to_csv(index=False)
                st.download_button(
                    label="Raw Data",
                    data = raw_csv_data,
                    file_name="DMR_Log_Raw_Data.csv",
                    mime="text/csv"
                )
            elif download_option == "Top 10 Oldest CLOSED DMR Data" and oldest_CLOSEDDMRs is not None:
                oldest_CLOSEDdata = oldest_CLOSEDDMRs.to_csv(index=False)
                st.download_button(
                    label="Top 10 Oldest CLOSED DMR Data",
                    data = oldest_CLOSEDdata,
                    file_name="Top10_Oldest_CLOSED_DMR_Data.csv",
                    mime="text/csv"
                )
            elif download_option == "Top 10 Oldest OPEN DMR Data" and oldest_OPENDMRs is not None:
                oldest_OPENdata = oldest_OPENDMRs.to_csv(index=False)
                st.download_button(
                    label="Top 10 Oldest OPEN DMR Data",
                    data = oldest_OPENdata,
                    file_name="Top10_Oldest_OPEN_DMR_Data.csv",
                    mime="text/csv"
                )
            elif download_option == "Histogram of DMRs Ages Chart" and filtered_data is not None:
                buf = BytesIO()
                pio.write_image(histogram_DMRs, buf, format='png')
                byte_im = buf.getvalue()

                st.download_button(
                    label ="Histogram of DMRs Ages Chart",
                    data=byte_im,
                    file_name="Histogram_DMRs_Age_Chart.png",
                    mime = "image/jpeg"
                )
            elif download_option=="DMR Counts Chart" and filtered_data is not None:
                image = BytesIO()
                pio.write_image(disposition_bar_chart, image, format = 'png')
                byte_im = image.getvalue()

                st.download_button(
                    label="DMR Counts Chart",
                    data = byte_im,
                    file_name="Disposition_Counts_Chart.png",
                    mime = 'image/jpeg'
                )
            elif download_option=="DMR Descriptive Statistics in Years":
                stats_data = df_combinded_stats_yrs.to_csv(index=False)
                st.download_button(
                    label="DMR Descriptive Statistics in YEARS",
                    data = stats_data,
                    file_name = "DMR_Descriptive_Statistics_YEARS.csv",
                    mime="text/csv"
                )
            elif download_option=="DMR Descriptive Statistics in Days":
                stats_data = df_combinded_stats_days.to_csv(index=False)
                st.download_button(
                    label="DMR Descriptive Statistics in DAYS",
                    data = stats_data,
                    file_name = "DMR_Descriptive_Statistics_DAYS.csv",
                    mime="text/csv"
                )

        #code that will check if a presentation for current date in already in the APCD Daily DMR Log folder
        presentation_title = f"DMR_Analysis_{formatted_current_date}.pptx"
        presentation_path = os.path.join(presentation_title).replace('/', '\\')
        presentation_exists = os.path.exists(presentation_path) #checking if presentation exists

        with st.sidebar:
            #checks if button is pressed
            if st.button("Generate PowerPoint for DMR Log"):
                #checks if the presentation_exists is true; meaning currents days presentation exists
                if presentation_exists == True:
                    st.success(f'DMR Analysis for ***{formatted_current_date}*** is **already** in "P:\Quality\DMR\APCD Daily DMR Log".')
                else:
                    st.info("Presentation of all data visulaizations in dashbaord will be put in PowerPoint.\
                    Function calls a class that will generate the PowerPoint.")
    else:
        st.warning('No data selected. Please ***upload*** the **"DMR MASTER LIST - USE THIS LOG"**.')

if __name__=='__main__':
    main()
