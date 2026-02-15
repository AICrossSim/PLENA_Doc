# PLENA Documentation

Documentation site for **PLENA** - A Programmable Long-context Efficient Neural Accelerator.

## View Documentation

Visit: https://AICrossSim.github.io/PLENA_Doc/

## Local Development

```bash
# Install dependencies
pip install mkdocs-material

# Serve locally (with live reload)
mkdocs serve

# Build static site
mkdocs build
```

## Deployment

The site automatically deploys to GitHub Pages when pushing to the `main` branch.

## Structure

```
doc/
├── index.md            # Home page
├── getting-started.md  # Installation and quick start
├── architecture.md     # System architecture overview
├── configuration.md    # Configuration reference
├── constraints.md      # Hardware constraints
├── models.md           # Model library
└── api/
    ├── interface.md    # Interface module API
    └── search.md       # Search module API
```
