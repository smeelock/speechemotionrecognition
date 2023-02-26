#cloud-config
users:
  - name: ${username}
    ssh_import_id:
      - gh:${username}
    lock_passwd: true
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash

runcmd:
  - echo "Hello, World!"