#setting the base image in Python 3.10.12
FROM python:3.10.12

#expose port 8501 for app to be run on
EXPOSE 8501

#set working directory
WORKDIR /app

#copy packages required from local requirements file to Docker image requirments
COPY requirements.txt ./requirements.txt
# Copying the PowerPoint template file into the Docker image
COPY APCD_PowerPoint_16x9_Template.pptx /app/
COPY Official_DMR_presentation_fileUpload.py /app/
COPY Official_DMR_Dash_fileUpload.py /app/

#run command line instructions specific to the Streamlit app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install pyodbc
RUN pip install kaleido  # Install kaleido package for image export
#RUN pip install python-pptx

#command to run Streamlit application
CMD streamlit run Official_DMR_Dash_fileUpload.py