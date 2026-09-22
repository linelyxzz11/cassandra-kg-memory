# Security Policy

## Sensitive information

Do not commit API keys, database passwords, service tokens, private datasets, local
credential notes, or `.env` files. Use `.env.example` to document variable names
without values.

If a credential appears in Git history, removing it from the current file is not
sufficient. Revoke or rotate the credential at the provider, then clean the history
only after identifying the exact affected commits and coordinating with repository
maintainers.

## Reporting

Do not open a public issue containing a secret or a private dataset sample. Use a
private GitHub security advisory when available, or contact the repository
maintainers through an established private channel. Include the affected path and
commit, but do not copy the secret into additional logs or messages.

## Research scope

The repository evaluates memory-serving behavior. It has not been audited as a
production authentication, authorization, or tenant-isolation system.
