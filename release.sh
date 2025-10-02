#!/bin/bash
# Automated Release Script for AI Image Tagger
# This script automates the entire release process

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AI Image Tagger - Automated Release${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if version is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Version number required${NC}"
    echo "Usage: ./release.sh <version> [prerelease]"
    echo "Example: ./release.sh 1.0.0"
    echo "Example: ./release.sh 1.0.0-beta prerelease"
    exit 1
fi

VERSION=$1
PRERELEASE=${2:-release}

# Validate version format
if [[ ! $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$ ]]; then
    echo -e "${RED}Error: Invalid version format${NC}"
    echo "Version should be in format: X.Y.Z or X.Y.Z-suffix"
    echo "Examples: 1.0.0, 1.2.3, 2.0.0-beta"
    exit 1
fi

TAG="v${VERSION}"

echo -e "${YELLOW}Version:${NC} $VERSION"
echo -e "${YELLOW}Tag:${NC} $TAG"
echo -e "${YELLOW}Type:${NC} $PRERELEASE"
echo ""

# Check if git is clean
if [[ -n $(git status -s) ]]; then
    echo -e "${YELLOW}⚠️  Warning: You have uncommitted changes${NC}"
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

echo -e "${GREEN}Step 1/5: Committing changes...${NC}"
git add .
git commit -m "Release $TAG" || echo "No changes to commit"

echo -e "${GREEN}Step 2/5: Pushing to GitHub...${NC}"
git push origin $(git branch --show-current)

echo -e "${GREEN}Step 3/5: Creating tag...${NC}"
git tag -a "$TAG" -m "Release $TAG"

echo -e "${GREEN}Step 4/5: Pushing tag...${NC}"
git push origin "$TAG"

echo -e "${GREEN}Step 5/5: Triggering GitHub Actions...${NC}"
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ Release $TAG initiated!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "GitHub Actions is now building:"
echo "  🐳 Docker image"
echo "  🐧 Linux executable"
echo "  🪟 Windows executable"
echo ""
echo "Monitor progress at:"
echo "  https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/actions"
echo ""
echo "Release will be available at:"
echo "  https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/releases/tag/$TAG"
echo ""
echo -e "${YELLOW}Note: Build takes approximately 20-30 minutes${NC}"
