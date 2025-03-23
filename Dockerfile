FROM python:3.11

# Create and set working directory
WORKDIR /app
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --default-timeout=3600 -r requirements.txt

# Set environment variables for Python and Streamlit
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_PORT=6996 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose Streamlit's default port
EXPOSE 6996

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port", "6996"]
