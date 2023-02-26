output "ssh_connect" {
  value = {
    username    = google_compute_instance.this.metadata.user
    ip          = google_compute_instance.this.network_interface[0].access_config[0].nat_ip
    ssh_connect = "ssh ${google_compute_instance.this.metadata.user}@${google_compute_instance.this.network_interface[0].access_config[0].nat_ip}"
  }
}
