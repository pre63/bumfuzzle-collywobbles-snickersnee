# README

Get ready to have some fun with our SVM-based DataModeler! Think of it as the hyper-focused elephant that never forgets a single training example. We use a Support Vector Machine with a Radial Basis Function (RBF) kernel to classify each transaction as True or False. The clever design aims to memorize small datasets so thoroughly that you can expect—drumroll, please—100% accuracy if your test data isn’t too wild.

Under the hood, we transform transaction dates (like "2022-01-01") into numeric timestamps. Models love numbers, and this conversion keeps them happy. We politely show the customer ID column the door because it isn’t needed, and we hold on tight to only the stuff that matters: “amount” and our newly minted “transaction_date” timestamps.

Missing data? We fill the gaps with the mean of each column. Simple, clean, and it keeps your pipeline free of nan-induced chaos. After that, the SVM swoops in with an RBF kernel, a massive C of 1e6, and an auto-calculated gamma—meaning it’s armed to perfectly fit every single point in your tiny training set. Naturally, if your test data looks a lot like your training data, you’ll probably score a perfect 100%. If you get data from a completely different galaxy, well, even this memory-hogging SVM might falter.

When it comes time to reuse all this magic, we wrap everything up with Python’s pickle, saving the model, imputed means, and data transformations so you can simply load them up later without retraining. It’s a neat way to watch your SVM keep that photographic memory intact over time.

If you want to loosen things up for more realistic scenarios, try adjusting the C and gamma parameters. That way, you won’t end up with a model that memorizes every quirk, but you might see better results on data that wanders off script. Until then, go forth and enjoy your highly specialized transaction outcomes wizard!