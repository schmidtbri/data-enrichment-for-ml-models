terraform {
  required_version = ">= 1.0.0, < 2.0.0"

  required_providers {
    digitalocean = {
      source = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
}

################################################
# Provider Configuration
################################################
provider "digitalocean" {}

provider "kubernetes" {
  host                   = digitalocean_kubernetes_cluster.cluster.endpoint
  token                  = digitalocean_kubernetes_cluster.cluster.kube_config[0].token
  cluster_ca_certificate = base64decode(
    digitalocean_kubernetes_cluster.cluster.kube_config[0].cluster_ca_certificate
  )
}

################################################
# Resources
################################################
resource "digitalocean_container_registry" "container_registry" {
  name                   = "${var.project_name}-registry"
  subscription_tier_slug = "basic"
}

resource "digitalocean_container_registry_docker_credentials" "registry_credentials" {
  registry_name = digitalocean_container_registry.container_registry.name
  write         = true
}

resource "digitalocean_vpc" "cluster_vpc" {
  name   = "${var.project_name}-vpc"
  region = var.region
}

data "digitalocean_kubernetes_versions" "k8s_versions" {
  version_prefix = "1.22."
}

resource "digitalocean_kubernetes_cluster" "cluster" {
  name     = "${var.project_name}-cluster"
  region   = var.region
  version  = data.digitalocean_kubernetes_versions.k8s_versions.latest_version
  vpc_uuid = digitalocean_vpc.cluster_vpc.id

  node_pool {
    name       = "${var.project_name}-cluster-worker-pool"
    size       = var.worker_type
    node_count = var.number_of_nodes
  }

  maintenance_policy {
    start_time  = "04:00"
    day         = "sunday"
  }
}

resource "kubernetes_secret" "cluster_registry_crendentials" {
  metadata {
    name = "docker-cfg"
  }

  data = {
    ".dockerconfigjson" = digitalocean_container_registry_docker_credentials.registry_credentials.docker_credentials
  }

  type = "kubernetes.io/dockerconfigjson"
}