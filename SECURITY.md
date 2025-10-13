# Security Policy

## Supported Versions

This section outlines the security support policy for the MARL Environments repository.

| Environment | Version | Supported Until |
|-------------|---------|------------------|
| DEM | All versions | Indefinite |
| HRG | All versions | Indefinite |
| MSFS | All versions | Indefinite |
| CM | All versions | Indefinite |
| SMAC | All versions | Indefinite |

## Reporting a Vulnerability

### How to Report

If you discover a security vulnerability in this repository, please report it privately to us before disclosing it publicly.

**Contact Information:**
- Email: sswunhuei@qq.com

### What to Include in Your Report

Please include the following information in your vulnerability report:

1. **Type of vulnerability** (e.g., code injection, denial of service, information disclosure)
2. **Affected versions** and components
3. **Detailed description** of the vulnerability
4. **Steps to reproduce** the issue
5. **Potential impact** if exploited
6. **Any proposed mitigation** (if available)

### Response Timeline

- **Initial response**: Within 48 hours
- **Detailed assessment**: Within 7 days
- **Patch development**: As soon as possible, typically within 14 days
- **Public disclosure**: After patch is available and coordinated with reporter

## Security Model

### Environment Sandboxing

All MARL environments run in isolated contexts with the following security boundaries:

```python
# Environment initialization with sandboxed parameters
env = create_dem_ctde_env(
    difficulty="normal",
    max_steps=100,  # Prevents infinite loops
    seed=None       # Controlled randomness
)
```

### Input Validation

The environments validate all inputs:

- **Action Space**: All actions are validated against allowed ranges
- **Configuration Parameters**: Invalid configurations raise exceptions
- **External Data**: File paths and network connections are restricted

### Output Sanitization

- **Observations**: Numeric arrays with type checking (float32, int64)
- **Rewards**: Bounded numeric values to prevent overflow
- **Game State**: Internal state is encapsulated and not directly accessible

## Known Security Considerations

### 1. Code Execution

**Risk Level**: Low
- Environment code execution is limited to defined interfaces
- No arbitrary code execution through environment actions
- Configuration files use safe data structures

**Mitigation**: All code runs in the same process as the main application

### 2. Resource Consumption

**Risk Level**: Medium
- Environments may consume CPU and memory resources
- No built-in resource limits for individual episodes

**Mitigation**:
- Use `max_steps` parameter to limit episode length
- Monitor resource usage during training
- Implement timeouts for environment operations

```python
# Example of safe resource usage
env = create_hrg_ctde_env(
    difficulty="easy_ctde",
    max_steps=50  # Reasonable limit
)
```

### 3. Random Number Generation

**Risk Level**: Low
- Uses numpy's RNG which is cryptographically insecure
- Predictable randomness could affect experiment reproducibility

**Mitigation**:
- Use seeds for reproducible experiments
- For security-critical applications, implement cryptographically secure RNG

### 4. File System Access

**Risk Level**: Low
- Environments do not access the file system directly
- Only read access to configuration files during initialization

**Mitigation**: File access is limited to configuration directory

### 5. Network Access

**Risk Level**: None
- Environments do not make network connections
- All communication happens through defined APIs

## Best Practices

### For Researchers

1. **Validate Inputs**: Always validate environment parameters
2. **Monitor Resources**: Set appropriate timeouts and memory limits
3. **Use Seeds**: Ensure reproducible experiments with random seeds
4. **Review Code**: Understand environment behavior before integration

```python
# Recommended usage pattern
def safe_environment_setup():
    env_config = {
        "difficulty": "normal",
        "max_steps": 100,
        "seed": 42  # Reproducible
    }

    try:
        env = create_dem_ctde_env(**env_config)
        return env
    except Exception as e:
        logger.error(f"Environment setup failed: {e}")
        return None
```

### For System Administrators

1. **Resource Monitoring**: Monitor CPU/memory usage during training
2. **Network Isolation**: Run experiments in isolated network environments
3. **User Permissions**: Run experiments with minimal required permissions
4. **Regular Updates**: Keep dependencies updated to latest secure versions

### For Developers

1. **Input Validation**: Validate all external inputs
2. **Error Handling**: Handle exceptions gracefully
3. **Logging**: Log security-relevant events
4. **Testing**: Include security tests in CI/CD pipeline

```python
# Example of secure coding practices
def validate_action(action, n_actions):
    """Validate action is within allowed range"""
    if not isinstance(action, (int, np.integer)):
        raise ValueError("Action must be integer")
    if action < 0 or action >= n_actions:
        raise ValueError(f"Action {action} out of range [0, {n_actions})")
    return True
```

## Dependency Security

This repository depends on the following packages with their security considerations:

| Package | Version | Security Notes |
|---------|---------|----------------|
| numpy | >=1.19.0 | Regular security updates, monitor CVEs |
| matplotlib | >=3.3.0 | Used for visualization, limited risk |
| gym (OpenAI) | optional | Legacy codebase, monitor for vulnerabilities |

**Recommended:**
```bash
pip install --upgrade numpy matplotlib
pip install --upgrade pip setuptools wheel
```

## Security Updates

### How We Handle Security Issues

1. **Assessment**: Security reports are assessed within 48 hours
2. **Development**: Patches are developed with priority over new features
3. **Testing**: Security patches undergo thorough testing
4. **Release**: Security fixes are released as soon as possible
5. **Disclosure**: Coordinated disclosure with security researchers

### Update Process

```bash
# To update to the latest secure version
git pull origin main
pip install -r requirements.txt --upgrade
```

## License and Usage

This security policy applies to all code in this repository. The security of these environments is the responsibility of:

- **Repository maintainers**: sswun
- **Users**: Researchers, developers, and organizations using these environments

### Security Guarantees

We provide:
- ✅ Regular security reviews
- ✅ Prompt vulnerability response
- ✅ Secure development practices
- ❌ No warranty of fitness for security-critical applications

### Usage Restrictions

**Do NOT use these environments for:**
- Security-critical production systems
- Real-time control of safety-critical equipment
- Financial trading or high-risk applications
- Any application where environment failure could cause harm

## Security Testing

### Automated Tests

```bash
# Run security-related tests
python -m pytest tests/security/ -v

# Check for known vulnerabilities
pip install safety
safety check
```

### Manual Security Checklist

- [ ] Validate all environment parameters
- [ ] Check resource consumption limits
- [ ] Verify random seed behavior
- [ ] Test error handling in edge cases
- [ ] Review dependencies for vulnerabilities
- [ ] Test with malicious inputs

## Contact Information

For security-related questions or vulnerability reports:

- **Primary Contact**: sswun@126.com
- **GitHub Issues**: Use security advisory feature for sensitive reports
- **Repository**: https://github.com/sswun/marlcoEnv

## Acknowledgments

We thank the security research community for:
- Responsible vulnerability disclosure
- Security best practices guidance
- Code reviews and security testing
- Contributing to safer open-source software

---

*This security policy is a living document and will be updated as new threats emerge and our security practices evolve.*

**Last Updated**: 2025-10-14
**Version**: 1.0
