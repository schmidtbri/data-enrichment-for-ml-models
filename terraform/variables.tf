variable "project_name" {
  description = "The name of the project."
  type        = string
}

variable "region" {
  description = "The region to use for the cluster."
  type        = string
  default     = "nyc1"
}

variable "number_of_nodes" {
  description = "The number of nodes to create in the cluster."
  type        = number
  default     = 3
}

variable "worker_type" {
  description = "The size of the droplet to use."
  type        = string
  default     = "s-2vcpu-2gb"
}