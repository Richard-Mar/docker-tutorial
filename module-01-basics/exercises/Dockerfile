FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple numpy
COPY script.py .
CMD ["python","script.py"]
