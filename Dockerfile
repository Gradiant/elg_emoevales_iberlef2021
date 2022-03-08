FROM python:3.7-slim

RUN addgroup --gid 1001 "elg" && \
      adduser --disabled-password --gecos "ELG User,,," \
      --home /elg --ingroup elg --uid 1001 elg && \
      mkdir /elg/workdir && \
      chmod -R +x /elg/

# Copy in our app, its requirements file and the entrypoint script
COPY --chown=elg:elg serve.py docker-entrypoint.sh /elg/
RUN chmod +x /elg/docker-entrypoint.sh && \
    chmod +x /elg/serve.py && chmod -R +x /elg/

# Everything from here down runs as the unprivileged user account
USER elg:elg
EXPOSE 8866
WORKDIR /elg/

# Create a Python virtual environment for the dependencies
RUN python -mvenv venv && \
	/elg/venv/bin/python -m pip install --upgrade pip && \
	venv/bin/pip --no-cache-dir install flask==2.0.2 flask_json==0.3.4 transformers==4.12.0 numpy==1.21.3 scipy==1.6.1 torch==1.10.0

ENV WORKERS=1
CMD ["/elg/venv/bin/python", "serve.py"]
