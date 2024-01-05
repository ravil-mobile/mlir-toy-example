#!/bin/bash

env_file="./.env"
echo "MUID=$(id -u)" > ${env_file}
echo "MGID=$(id -g)" >> ${env_file}
echo "USERNAME=$(whoami)" >> ${env_file}
