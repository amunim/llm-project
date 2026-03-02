# Project Overview

This document summarizes the project goal, data handling, and model choice.

## Goal
Create a privacy-aware, LLM-powered assistant that can answer customer queries using a curated dataset of bank product information.

## Data Handling
- Parse Excel/JSON/CSV/text
- Clean and normalize text
- Apply anonymization (regex masking + optional NER)
- Produce LLM-ready chunks

## Model Choice
Using **Gemma 3 1B** (<= 6B parameters) for its small footprint, solid instruction-following behavior, and suitability for prompt engineering or LoRA fine-tuning.
