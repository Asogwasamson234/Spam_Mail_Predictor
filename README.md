# Spam Mail Detection System

A machine learning-based system to detect whether an email or SMS message is spam or legitimate (ham). The system
combines two datasets and provides both GUI and command-line interfaces for easy use.

## Table of Contents

- [Overview](#overview)
- [Dependencies and Installation](#dependencies-and-installation)
- [Datasets Needed](#datasets-needed)
- [Work Flow/Architecture](#work-flowarchitecture)
- [Application/Uses](#applicationuses)
- [Future Improvements](#future-improvements)
- [Disclaimer](#disclaimer)

## Overview

This project implements a spam detection system using **Logistic Regression** with **TF-IDF** feature extraction. The
system is trained on two datasets:

1. **Enron Spam Dataset** - Email data
2. **SMS Spam Collection** - SMS messages

The trained model can predict whether a given message is spam or ham with high accuracy. The system provides:

- A **GUI application** for user-friendly interaction
- A **command-line interface** for quick testing
- **Confidence scores** for predictions
- **Performance metrics** and statistics

## Dependencies and Installation

### Required Packages

```bash
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

# Datasets Needed

Dataset Locations
Place the following datasets in the data_sets folder:

#### 1 Enron Spam Dataset

* Path:
  C:\Users\hp\Documents\Projects\spam_mail_prediction\data_sets\enron_spam_data.csv

* Contains email messages labeled as 'spam' or 'ham'

#### 2 SMS Spam Collection

- Path:
  C:\Users\hp\Documents\Projects\spam_mail_prediction\data_sets\spam_original.csv

- Contains SMS messages labeled as 'spam' or 'ham'

### Dataset Format

Both datasets should have the following structure:

- Enron dataset: Columns - Message, Spam/Ham

- SMS dataset: Columns - v1 (label), v2 (message)

### Dataset Statistics

- Combined dataset contains over 5,000 messages

- Balanced mix of spam and ham messages

- Used for training (80%) and testing (20%)

# Work Flow/Architecture

![](https://viewer.diagrams.net/?tags=%7B%7D&lightbox=1&highlight=0000ff&edit=_blank&layers=1&nav=1&dark=auto#R%3Cmxfile%3E%3Cdiagram%20name%3D%22Page-1%22%20id%3D%22pRIeWYRSu3OTkmSqJit1%22%3E7VnbbuIwEP0apO4DKyAkwGPLrZVabVXQ7vbRTUyw5NhZx0Do1%2B84ccit3FSSVlUfWuzjmXEyc2YycRrG0AunAvnLB%2B5g2ui0nLBhjBqdTr%2FXgf8K2MZA2zQHMeIK4mgsBWbkFWuwpdEVcXCQE5ScU0n8PGhzxrAtcxgSgm%2FyYgtO87v6yMUlYGYjWkb%2FEEcu9X2ZrRS%2FxcRdJju3W3rFQ4mwBoIlcvgmAxnjhjEUnMt45IVDTJXzEr%2FEepM9q7sLE5jJUxS29Nf62RncTRnhwXXz9Xra85pvWNFQILeJDzZLIvHMR7aabyDODeNmKT0KszYMUeDHnl%2BQEMNmN7GBNaIrbaDRsahUSl0YuWo0QhKBzD1HDmFusv4iknX4RZ7ah70E6gcxpwiVlWzuvRCWsQfOSLdMQIesT4Byl5IYe0OuiEZ3joXEYcabOiBTzD0sxRZEllnOdDVDNinBdpjOG0NPkaazuzOVRhwGOuhnEKBXBQEQJS6DuQ02sQBA8BVz1FIkfwI%2F7pi%2FkiCkeXI1e5jBzxyHAE7GHiL0x54gVxSAZrtXVQj6H5SDCbLgTLl6EdsG9N9K1aSb31g4iKEUSBQfBbdxEMSJ9mZOgh9iq6VkOSkfV7SIUJIgQ4oRU0%2BBiAs7u6lAWeUJe3yNI39yX4WZCyc4Ufeeb7Bo2ijABxUALF90TbWhuuIw%2BCBmTjCSK6EiNg6lQLYknJUL83wCAk34uxtN6i0G3W5VHm9r%2F2Kn1JaUY8BXQqfswYd7JlaYOdeqKVK1mSJIYTsfrLROt2CGQyL%2FqvFPU8%2BetZwaj8KM2GibTBi4IaOkps%2FZtVQtmiV6urVDwsXyeK3MsedYRJNeTWCKJFnn%2FZqJsnkgqHqHR06iWpnQySpkolUwEYdIa2V7s4KhXbu7LcwTQ7FjSoYiju1u%2Bx2065xBuy9Ep8GnopM1gJvMdX3FnuNUPpUsdYvPiKoJZXxJQn0SovQKvWkxuKfSpHf4JaNqjnSraC9cxZv9QSu2GnOBCPQWwzkO5PE3zJlPiSzDV33wHCQc%2BKGl3sLNml9OrEMUel8%2FYn7ncY0F3%2BwbF8lky7TqTWXrmyY10qQ3uEzB71tGvTTpfdOkRpoMLkSTdqtunnzUkdg9d0kggV9HO4En7Ap1BhYdTByR1V9F6uwI%2BsU30ct1BJUcCp19QPwosEP2HAxd3SIP7g2292puxCBTOmf7HabpV6A4hdJvacb4Pw%3D%3D%3C%2Fdiagram%3E%3C%2Fmxfile%3E)

![spam_detection.drawio.drawio.svg](C:\Users\hp\Downloads\spam_detection.drawio.svg)