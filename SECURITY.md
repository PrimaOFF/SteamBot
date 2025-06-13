# Security Configuration Guide

## Overview

This document provides security configuration guidelines for the CS2 Float Checker production deployment.

## Environment Variables Setup

### 1. Create Environment File

Copy the example environment file and customize it:

```bash
cp .env.example .env
```

### 2. Generate Secure Passwords

Use the following commands to generate secure passwords:

```bash
# PostgreSQL password
openssl rand -base64 32

# Redis password  
openssl rand -base64 32

# Grafana admin password
openssl rand -base64 32

# API secret key
openssl rand -base64 64

# JWT secret key
openssl rand -base64 64
```

### 3. Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `STEAM_API_KEY` | Steam Web API key | `ABCD1234...` |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token | `123456:ABC-DEF...` |
| `TELEGRAM_CHAT_ID` | Telegram chat ID | `-123456789` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `secure_db_password_123` |
| `REDIS_PASSWORD` | Redis password | `secure_redis_password_456` |
| `GRAFANA_ADMIN_PASSWORD` | Grafana admin password | `secure_grafana_password_789` |
| `API_SECRET_KEY` | API authentication key | `secure_api_key_xyz` |

### 4. File Permissions

Ensure proper file permissions for the `.env` file:

```bash
chmod 600 .env
chown root:root .env
```

## Security Best Practices

### 1. Password Requirements

- Minimum 16 characters for production passwords
- Use a combination of uppercase, lowercase, numbers, and symbols
- Rotate passwords regularly (every 90 days)
- Never reuse passwords across services

### 2. Network Security

- Use internal Docker networks for service communication
- Limit external port exposure
- Enable firewall rules for production deployment
- Use TLS/SSL for all external communications

### 3. Database Security

- Enable PostgreSQL SSL connections in production
- Use dedicated database users with minimal privileges
- Regular database backups with encryption
- Monitor database access logs

### 4. Redis Security

- Always use password authentication
- Disable dangerous commands in production
- Use Redis ACLs for fine-grained access control
- Monitor Redis access patterns

### 5. API Security

- Use strong API keys for authentication
- Implement rate limiting
- Log all API access attempts
- Regular security audits

## Production Deployment Security

### 1. Docker Security

```bash
# Run containers as non-root user
# Enable Docker Content Trust
export DOCKER_CONTENT_TRUST=1

# Scan images for vulnerabilities
docker scan cs2-float-checker:latest
```

### 2. Monitoring and Alerting

- Enable security monitoring in Grafana
- Set up alerts for failed authentication attempts
- Monitor unusual network traffic patterns
- Regular security log reviews

### 3. Backup Security

- Encrypt all backup files
- Store backups in secure, separate locations
- Test backup restoration procedures regularly
- Implement backup retention policies

## Incident Response

### 1. Security Breach Response

1. Immediately rotate all compromised credentials
2. Review access logs for unauthorized activity
3. Update all affected services
4. Notify stakeholders
5. Document the incident

### 2. Password Rotation Procedure

```bash
# 1. Generate new passwords
NEW_POSTGRES_PASSWORD=$(openssl rand -base64 32)
NEW_REDIS_PASSWORD=$(openssl rand -base64 32)

# 2. Update .env file
sed -i "s/POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$NEW_POSTGRES_PASSWORD/" .env
sed -i "s/REDIS_PASSWORD=.*/REDIS_PASSWORD=$NEW_REDIS_PASSWORD/" .env

# 3. Restart services
docker-compose restart postgres redis

# 4. Verify connections
docker-compose logs postgres redis
```

## Compliance and Auditing

### 1. Regular Security Audits

- Monthly password rotation
- Quarterly security assessments
- Annual penetration testing
- Continuous vulnerability scanning

### 2. Logging Requirements

- Authentication attempts (successful and failed)
- Administrative actions
- Database access patterns
- API usage statistics

### 3. Data Protection

- Encrypt sensitive data at rest
- Use TLS for data in transit
- Implement data retention policies
- Regular data classification reviews

## Emergency Contacts

In case of security incidents:

1. **Security Team**: security@yourcompany.com
2. **System Administrator**: admin@yourcompany.com
3. **Emergency Hotline**: +1-XXX-XXX-XXXX

## Security Updates

This document should be reviewed and updated:

- After any security incidents
- Following major system updates
- At least quarterly
- When new services are added

---

**Last Updated**: June 2025  
**Version**: 1.0  
**Review Date**: September 2025