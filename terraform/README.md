# Terraform 

This terraform module is able to:

- create a docker registry
- create a VPC
- create a kubernetes cluster with 3 nodes
- add the docker registry credentials to the local docker daemon
- add the docker registry credentials to the kubernetes cluster so images can be accessed from inside the cluster

## Pre-requisites

* You must have [Terraform](https://www.terraform.io/) installed on your computer. 
* You must have an DigitalOcean account.

## Quick start

**Please note that this example will deploy real resources into your account, and we are not responsible for any
charges you may incur.** 

Configure your [Digital Ocean Token](https://docs.digitalocean.com/reference/api/create-personal-access-token/) as an 
environment variable:

```bash
export DIGITALOCEAN_TOKEN=(your access token)
```

View the plan:
```bash
terraform init
terraform plan
```

Deploy the code:
```bash
terraform apply
```

Clean up when you're done:

```bash
terraform destroy
```
