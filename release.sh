#!/bin/bash
# Automated Release Script for AI Image Captioner
# Builds ALL configurations from version.json for a single release

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AI Image Captioner - Release Manager${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to show usage
show_usage() {
    echo "Usage: ./release.sh -v VERSION [OPTIONS]"
    echo ""
    echo "This script builds ALL configurations and releases them under a single tag."
    echo ""
    echo "Options:"
    echo "  -v, --version VERSION    Set release version (e.g., 1.0.2) [REQUIRED]"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./release.sh -v 1.0.2    # Build all configs and release as v1.0.2"
    echo ""
    echo "All configurations from version.json will be built:"
    jq -r '.build_configs | to_entries[] | "  - \(.key): Python \(.value.python), CUDA \(.value.cuda_version_display)"' version.json
    echo ""
    echo "This creates executables like:"
    echo "  - ai-image-captioner-windows-python310-cuda118.zip"
    echo "  - ai-image-captioner-linux-python310-cuda124.tar.gz"
    echo "  - etc. (all configs × 2 platforms)"
    exit 0
}

# Parse arguments
VERSION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_usage
            ;;
    esac
done

# Check if version is provided
if [ -z "$VERSION" ]; then
    echo -e "${RED}Error: Version required${NC}"
    echo ""
    show_usage
fi

# Validate version format
if [[ ! $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$ ]]; then
    echo -e "${RED}Error: Invalid version format${NC}"
    echo "Version should be: X.Y.Z or X.Y.Z-suffix"
    echo "Examples: 1.0.0, 1.2.3, 2.0.0-beta"
    exit 1
fi

TAG="v${VERSION}"

# Show what will be built
NUM_CONFIGS=$(jq '.build_configs | length' version.json)
echo -e "${YELLOW}Release Configuration:${NC}"
echo "  Version Tag:   $TAG"
echo "  Configs:       $NUM_CONFIGS configurations"
echo "  Platforms:     Windows + Linux"
echo "  Total Builds:  $((NUM_CONFIGS * 2)) executables + $NUM_CONFIGS Docker images"
echo ""
echo -e "${YELLOW}Configurations to build:${NC}"
jq -r '.build_configs | to_entries[] | "  ✓ \(.key): Python \(.value.python), CUDA \(.value.cuda_version_display)"' version.json
echo ""

# Check if git is clean
if [[ -n $(git status -s) ]]; then
    echo -e "${YELLOW}⚠️  Uncommitted changes detected:${NC}"
    echo ""
    git status -s
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Release cancelled${NC}"
        exit 1
    fi
fi

# Check if tag already exists
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo -e "${RED}Error: Tag $TAG already exists${NC}"
    echo "To delete: git tag -d $TAG && git push origin :refs/tags/$TAG"
    exit 1
fi

echo -e "${GREEN}Step 1/4: Committing changes...${NC}"
git add .
git commit -m "Release $TAG" || echo "No changes to commit"

echo -e "${GREEN}Step 2/4: Creating tag...${NC}"
git tag -a "$TAG" -m "Release $TAG"

echo -e "${GREEN}Step 3/4: Pushing to GitHub...${NC}"
git push origin $(git branch --show-current)

echo -e "${GREEN}Step 4/4: Pushing tag (triggers GitHub Actions)...${NC}"
git push origin "$TAG"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ Release $TAG initiated!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "GitHub Actions will build ALL configurations:"
echo ""
jq -r '.build_configs | to_entries[] | "  • \(.key) → Windows + Linux executables + Docker image"' version.json
echo ""
echo "Monitor progress:"
echo "  $(git config --get remote.origin.url | sed 's/.*github.com[:/]\(.*\)\.git/https:\/\/github.com\/\1\/actions/')"
echo ""
echo "Release page (artifacts available when build completes):"
echo "  $(git config --get remote.origin.url | sed 's/.*github.com[:/]\(.*\)\.git/https:\/\/github.com\/\1\/releases\/tag\/'"$TAG"'/')"
echo ""
echo -e "${YELLOW}Estimated build time: ~20-30 minutes${NC}"
echo -e "${YELLOW}All configurations build in parallel for faster completion${NC}"
