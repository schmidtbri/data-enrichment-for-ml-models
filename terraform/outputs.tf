output "registry_endpoint" {
  value       = digitalocean_container_registry.container_registry.endpoint
  description = "Endpoint of the container registry created."
}

output "kubernetes_cluster_id" {
  value       = digitalocean_kubernetes_cluster.cluster.id
  description = "Id of the cluster created."
}