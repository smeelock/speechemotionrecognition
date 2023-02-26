#cloud-config
groups:
  - docker
users:
  - name: ${username}
    ssh_import_id:
      - gh:${username}
    lock_passwd: true
    sudo: ALL=(ALL) NOPASSWD:ALL
    groups: [docker] # for sudoless docker
    shell: /bin/bash

runcmd:
  # Install docker using convenience script
  # https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script
  # Not suitable for production but good and simple enough for our use case
  - curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
  - sh /tmp/get-docker.sh
  - rm -rf /tmp/get-docker.sh
  # Note: docker compose is now installed with docker

  # Install kubectl
  # https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/
  - curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
  - install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
  - kubectl completion bash | sudo tee /etc/bash_completion.d/kubectl > /dev/null

  # Install Helm
  - curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

  # Install Kind
  - curl -L https://kind.sigs.k8s.io/dl/v0.15.0/kind-linux-amd64 -o /usr/local/bin/kind
  - chmod +x /usr/local/bin/kind