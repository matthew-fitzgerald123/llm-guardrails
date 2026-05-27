variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project" {
  type    = string
  default = "llm-guardrails"
}

variable "vpc_id" {
  type        = string
  description = "VPC ID from P1 (financial-sentiment-llm) deployment"
}

variable "public_subnet_ids" {
  type        = list(string)
  description = "Public subnet IDs from P1 VPC"
}

variable "database_url" {
  type      = string
  sensitive = true
}

variable "redis_url" {
  type    = string
  default = "redis://localhost:6379"
}

variable "upstream_url" {
  type        = string
  description = "URL of the P5 llm-agent service"
  default     = "http://llm-agent-alb.example.com"
}

variable "rate_limit_tiers" {
  type    = string
  default = "free:20,standard:100,premium:500"
}
