#!/bin/bash

if ! type cloud-init > /dev/null 2>&1 ; then
  echo "Ran - `date`" >> /root/startup
  sleep 30
  sudo apt-get update -yq && apt-get install -yq \
     cloud-init

  if [ $? == 0 ]; then
    echo "Ran - Success - `date`" >> /root/startup
    systemctl enable cloud-init
  else
    echo "Ran - Fail - `date`" >> /root/startup
  fi

  # Reboot either way
  reboot
fi