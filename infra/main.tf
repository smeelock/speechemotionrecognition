locals {
  project_name = "speechemotionrecognition"
  project_id   = "adroit-crow-376514"
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

resource "google_compute_instance" "default" {
  name         = "${local.project_name}-instance"
  machine_type = "e2-micro"

  tags = ["tsinghua", "speechemotionrecognition"]
  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
    }

  }

  network_interface {
    network = "default"
    access_config {
      network_tier = "STANDARD"
    }
  }
}
