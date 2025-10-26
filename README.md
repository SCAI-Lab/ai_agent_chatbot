# AI Agent Chatbot Project

An intelligent chatbot system with long-term memory capabilities, featuring voice interaction and personality modeling.

## Project Overview

This repository houses a complete AI chatbot solution with advanced memory management, designed to deliver personalized conversational experiences through context retention and learning from past interactions.

## Project Components

### [chatbot/](chatbot/)
A custom AI chatbot optimized for **Nvidia Orin 64GB** platform, featuring sophisticated memory integration and natural conversation capabilities.

**Core Capabilities:**
- Persistent long-term memory for personalized interactions
- Dynamic personality and emotion modeling
- Real-time voice interaction support
- Hardware-optimized performance for Nvidia Orin

### [memobase/](memobase/)
Backend memory management system providing robust infrastructure for conversational data storage, retrieval, and organization.

**Documentation:** https://github.com/memodb-io/memobase

### [memobase-inspector/](memobase-inspector/)
Web-based frontend for visualizing and managing the memory database, offering intuitive tools for inspection and interaction.

**Documentation:** https://github.com/memodb-io/memobase-inspector

## Getting Started

### Prerequisites
- Python 3.10
- Required dependencies (see individual component READMEs)

### Installation

1. **Set up Memobase (Memory System)**

   Follow the official installation guide:
   ```
   https://github.com/memodb-io/memobase
   ```

2. **Set up Memobase Inspector (Optional)**

   For memory visualization and management interface:
   ```
   https://github.com/memodb-io/memobase-inspector
   ```

3. **Deploy the Chatbot**

   See detailed setup instructions in the [chatbot README](chatbot/README.md)

## License

Each component maintains its own license. Please check individual directories for details.
