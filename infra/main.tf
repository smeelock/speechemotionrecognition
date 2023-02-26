locals {
  project_name = "speechemotionrecognition"
  project_id   = "adroit-crow-376514"
  username     = "smeelock"
}

terraform {
  required_providers {
    google = {
      version = "~> 2.13.0"
    }
  }
  required_version = "~> 0.12.29"
}

provider "google" {
  project = local.project_id
  region  = "us-central1"
  zone    = "us-central1-c"
}

resource "google_compute_network" "default" {
  name = "${local.project_name}-network"
}

resource "google_compute_firewall" "default-allow-ssh" {
  name    = "${local.project_name}-firewall"
  network = google_compute_network.default.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

resource "google_compute_instance" "this" {
  name         = "${local.project_name}-${local.username}-instance"
  machine_type = "e2-micro"

  tags = ["tsinghua", "speechemotionrecognition", local.username]
  boot_disk {
    initialize_params {
      image = "ubuntu-os-pro-cloud/ubuntu-pro-2204-lts"
    }

  }

  network_interface {
    network = google_compute_network.default.name
    access_config {
      network_tier = "STANDARD"
    }
  }

  metadata = {
    user-data      = templatefile("templates/userdata.yml.tpl", { username = local.username })
    startup-script = file("startup.sh")
    user           = local.username
  }
}
