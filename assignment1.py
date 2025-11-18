import gzip
import math
import random
import string
from collections import defaultdict

try:
    # High-end text models for category prediction 
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    # High-end rating model (Surprise SVD) if available
    from surprise import SVD as SurpriseSVD
    from surprise import Dataset as SurpriseDataset
    from surprise import Reader as SurpriseReader
    SURPRISE_AVAILABLE = True
except Exception:
    SURPRISE_AVAILABLE = False


###############################################################################
# Helpers
###############################################################################

random.seed(0)


def readGz(path):
    """Read gzipped JSON lines file."""
    for l in gzip.open(path, "rt", encoding="utf-8"):
        # The provided data is Python-literal JSON; eval is used in the baseline.
        yield eval(l)


def readCSV(path):
    """Read gzipped CSV file, skipping header."""
    f = gzip.open(path, "rt")
    f.readline()  # Skip header
    for l in f:
        yield l.strip().split(",")


punct_translator = str.maketrans("", "", string.punctuation)


def tokenize(text):
    """Lowercase, strip punctuation, split, and drop very short tokens/digits."""
    text = text.lower().translate(punct_translator)
    return [w for w in text.split() if len(w) > 2 and not w.isdigit()]


###############################################################################
# TASK 1: RATING PREDICTION
# Prefer: Surprise SVD if available, else MF with biases + latent factors
###############################################################################

print("Loading training data for rating prediction...")

interactions = []  # (user, book, rating)
userRatings = defaultdict(list)
itemRatings = defaultdict(list)

for user, book, r in readCSV("train_Interactions.csv.gz"):
    r = int(r)
    interactions.append((user, book, r))
    userRatings[user].append(r)
    itemRatings[book].append(r)

allRatings = [r for _, _, r in interactions]
globalAverage = sum(allRatings) / len(allRatings)

print(f"Global average rating: {globalAverage:.3f}")
print(f"Number of interactions: {len(interactions)}")
print(f"Number of users: {len(userRatings)}")
print(f"Number of books: {len(itemRatings)}")

# NOTE: Surprise SVD hurt public MSE (worse than our manual MF),
# so we force the fallback MF path for the final submission.
if False and SURPRISE_AVAILABLE:
    # ---------------------------------------------------------------------
    # High-end model: Surprise SVD (matrix factorization with biases)
    # ---------------------------------------------------------------------
    print("Using Surprise SVD for rating prediction...")

    # Write interactions to a temporary tsv file for Surprise
    tmp_path = "surprise_train_interactions.tsv"
    with open(tmp_path, "w") as f:
        for u, b, r in interactions:
            f.write(f"{u}\t{b}\t{r}\n")

    reader = SurpriseReader(line_format="user item rating", sep="\t")
    data = SurpriseDataset.load_from_file(tmp_path, reader=reader)
    trainset = data.build_full_trainset()

    algo = SurpriseSVD(
        n_factors=50,
        reg_all=0.02,
        lr_all=0.005,
        n_epochs=40,
        biased=True,
    )
    algo.fit(trainset)

    def predict_rating(u, b):
        return algo.predict(u, b).est

else:
    # ---------------------------------------------------------------------
    # Fallback: manual MF with user/item biases + latent factors
    # ---------------------------------------------------------------------
    print("Surprise not available; using manual MF with biases + latent factors.")

    # Build index mappings for latent factors
    user_index = {}
    item_index = {}
    for u, _, _ in interactions:
        if u not in user_index:
            user_index[u] = len(user_index)
    for _, b, _ in interactions:
        if b not in item_index:
            item_index[b] = len(item_index)

    n_users = len(user_index)
    n_items = len(item_index)

    print(f"Indexed users: {n_users}, indexed items: {n_items}")

    # Train regularized user and item biases + latent factors with SGD
    bu = defaultdict(float)
    bi = defaultdict(float)

    # Latent factor dimension
    K = 20
    user_factors = [[(0.1 * (random.random() - 0.5)) for _ in range(K)] for _ in range(n_users)]
    item_factors = [[(0.1 * (random.random() - 0.5)) for _ in range(K)] for _ in range(n_items)]

    gamma = 0.01  # learning rate
    reg = 0.02    # regularization strength
    epochs = 15

    for epoch in range(epochs):
        random.shuffle(interactions)
        se = 0.0
        for u, i, r in interactions:
            ui = user_index[u]
            ii = item_index[i]

            # Dot product for latent factors
            pu = user_factors[ui]
            qi = item_factors[ii]
            dot = 0.0
            for k in range(K):
                dot += pu[k] * qi[k]

            pred = globalAverage + bu[u] + bi[i] + dot
            err = r - pred

            se += err * err

            # Update biases
            bu[u] += gamma * (err - reg * bu[u])
            bi[i] += gamma * (err - reg * bi[i])

            # Update latent factors
            for k in range(K):
                p_uk = pu[k]
                q_ik = qi[k]
                pu[k] += gamma * (err * q_ik - reg * p_uk)
                qi[k] += gamma * (err * p_uk - reg * q_ik)

        rmse = math.sqrt(se / len(interactions))
        print(f"Epoch {epoch + 1}/{epochs} - training RMSE (with factors): {rmse:.4f}")

    def predict_rating(u, b):
        """Predict rating using global mean + learned biases + latent factors (if available)."""
        base = globalAverage + bu.get(u, 0.0) + bi.get(b, 0.0)
        ui = user_index.get(u)
        ii = item_index.get(b)
        if ui is None or ii is None:
            return base
        pu = user_factors[ui]
        qi = item_factors[ii]
        dot = 0.0
        for k in range(len(pu)):
            dot += pu[k] * qi[k]
        return base + dot


print("Generating rating predictions...")
with open("predictions_Rating.csv", "w") as predictions:
    for l in open("pairs_Rating.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u, b = l.strip().split(",")

        pred = predict_rating(u, b)
        pred = max(0.0, min(5.0, pred))  # clip to valid range
        predictions.write(u + "," + b + "," + str(pred) + "\n")

print("Rating predictions complete!")


###############################################################################
# TASK 2: READ PREDICTION
# Logistic regression on popularity + activity + rating-based features
###############################################################################

print("\nPreparing data for read prediction...")

userBooks = defaultdict(set)
bookCount = defaultdict(int)
bookRatingSum = defaultdict(float)
bookRatingCount = defaultdict(int)

for u, b, r in interactions:
    userBooks[u].add(b)
    bookCount[b] += 1
    bookRatingSum[b] += r
    bookRatingCount[b] += 1

allBooks = list(bookCount.keys())
userActivity = {u: len(books) for u, books in userBooks.items()}
bookAvgRating = {
    b: bookRatingSum[b] / bookRatingCount[b] for b in bookRatingSum
}


def build_features(u, b):
    """Return feature vector [1, f1, f2, f3, f4] for logistic regression."""
    # Book popularity (log-scaled)
    pop = bookCount.get(b, 0)
    f1 = math.log(1.0 + pop)

    # User activity (log-scaled)
    act = userActivity.get(u, 0)
    f2 = math.log(1.0 + act)

    # Book average rating (centered)
    avg_r = bookAvgRating.get(b, globalAverage)
    f3 = avg_r - globalAverage

    # Predicted rating (centered)
    pr = predict_rating(u, b)
    f4 = pr - globalAverage

    return [1.0, f1, f2, f3, f4]


print("Building training samples for read prediction (with negative sampling)...")

samples = []  # (features, label)

for u, b, _ in interactions:
    # Positive sample
    samples.append((build_features(u, b), 1.0))

    # One negative sample: random book not read by this user (approximate)
    for _ in range(3):  # try a few times to find an unseen book
        neg_b = random.choice(allBooks)
        if neg_b not in userBooks[u]:
            samples.append((build_features(u, neg_b), 0.0))
            break

print(f"Training samples (including negatives): {len(samples)}")


def sigmoid(x):
    if x < -20:
        return 0.0
    if x > 20:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


# Train logistic regression: p(read | features)
w = [0.0] * 5  # 4 features + bias
lr = 0.01
reg_lr = 0.0001
epochs_lr = 8

for epoch in range(epochs_lr):
    random.shuffle(samples)
    total_loss = 0.0
    correct = 0
    for x, y in samples:
        z = sum(wi * xi for wi, xi in zip(w, x))
        p = sigmoid(z)
        # Logistic loss
        total_loss += -(y * math.log(p + 1e-9) + (1 - y) * math.log(1 - p + 1e-9))

        # Classification accuracy on-the-fly
        pred_label = 1.0 if p >= 0.5 else 0.0
        if pred_label == y:
            correct += 1

        # Gradient update
        grad = (y - p)
        for i in range(len(w)):
            w[i] += lr * (grad * x[i] - reg_lr * w[i])

    avg_loss = total_loss / len(samples)
    acc = correct / len(samples)
    print(f"Read LR epoch {epoch + 1}/{epochs_lr} - loss: {avg_loss:.4f}, acc: {acc:.4f}")


print("Scoring read pairs and thresholding for exactly 50% positives...")
pairs = []
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        continue
    u, b = l.strip().split(",")
    pairs.append((u, b))

scores = []
for idx, (u, b) in enumerate(pairs):
    x = build_features(u, b)
    z = sum(wi * xi for wi, xi in zip(w, x))
    scores.append((idx, z))

# Sort by score descending and assign top 50% as positive
scores.sort(key=lambda t: t[1], reverse=True)
N = len(scores)
num_pos = N // 2
pred_labels = [0] * N
for rank in range(num_pos):
    idx, _ = scores[rank]
    pred_labels[idx] = 1

print("Generating read predictions (balanced 50/50)...")
with open("predictions_Read.csv", "w") as predictions:
    first = True
    pos = 0
    for l in open("pairs_Read.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u, b = l.strip().split(",")
        pred = pred_labels[pos]
        predictions.write(u + "," + b + "," + str(pred) + "\n")
        pos += 1

print("Read predictions complete!")


###############################################################################
# TASK 3: CATEGORY PREDICTION
# Prefer: TF-IDF + linear classifier (sklearn) if available.
# Fallback: custom dictionary logistic regression.
###############################################################################

print("\nLoading training data for category prediction...")

catDict = {
    "children": 0,
    "comics_graphic": 1,
    "fantasy_paranormal": 2,
    "mystery_thriller_crime": 3,
    "young_adult": 4,
}

num_classes = len(catDict)

train_reviews = list(readGz("train_Category.json.gz"))
print(f"Training documents: {len(train_reviews)}")

if SKLEARN_AVAILABLE:
    # ---------------------------------------------------------------------
    # High-end model: TF-IDF (unigrams+bigrams) + LogisticRegression
    # ---------------------------------------------------------------------
    print("Using sklearn TF-IDF + LogisticRegression for category prediction (with hyperparameter search)...")

    # Import here to avoid errors when sklearn is absent
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    texts = [r["review_text"] for r in train_reviews]
    y = [r["genreID"] for r in train_reviews]

    # Hyperparameter search over vectorizer + classifier + C
    vec_configs = [
        {"ngram_range": (1, 2), "max_features": 50000},
        {"ngram_range": (1, 3), "max_features": 80000},
    ]
    C_grid_logreg = [0.5, 1.0, 2.0, 3.0]
    C_grid_svc = [0.5, 1.0, 2.0]

    overall_best = {
        "acc": -1.0,
        "C": None,
        "model": None,  # "logreg" or "linsvc"
        "ngram_range": None,
        "max_features": None,
    }

    for vc in vec_configs:
        print(
            f"Trying vectorizer config: ngram_range={vc['ngram_range']}, "
            f"max_features={vc['max_features']}"
        )
        vectorizer_tmp = TfidfVectorizer(
            lowercase=True,
            ngram_range=vc["ngram_range"],
            max_features=vc["max_features"],
            min_df=3,
            max_df=0.8,
        )
        X_all = vectorizer_tmp.fit_transform(texts)
        Xtr, Xva, ytr, yva = train_test_split(
            X_all, y, test_size=0.2, random_state=0
        )

        # LogisticRegression grid
        for C in C_grid_logreg:
            clf_tmp = LogisticRegression(
                C=C,
                penalty="l2",
                solver="lbfgs",
                max_iter=5000,
                n_jobs=-1,
            )
            clf_tmp.fit(Xtr, ytr)
            pred_va = clf_tmp.predict(Xva)
            acc_va = accuracy_score(yva, pred_va)
            print(
                f"  LogReg C={C} -> val acc: {acc_va:.4f} "
                f"(ngrams={vc['ngram_range']}, max_features={vc['max_features']})"
            )
            if acc_va > overall_best["acc"]:
                overall_best["acc"] = acc_va
                overall_best["C"] = C
                overall_best["model"] = "logreg"
                overall_best["ngram_range"] = vc["ngram_range"]
                overall_best["max_features"] = vc["max_features"]

        # LinearSVC grid (if available)
        for C in C_grid_svc:
            clf_tmp = LinearSVC(C=C)
            clf_tmp.fit(Xtr, ytr)
            pred_va = clf_tmp.predict(Xva)
            acc_va = accuracy_score(yva, pred_va)
            print(
                f"  LinearSVC C={C} -> val acc: {acc_va:.4f} "
                f"(ngrams={vc['ngram_range']}, max_features={vc['max_features']})"
            )
            if acc_va > overall_best["acc"]:
                overall_best["acc"] = acc_va
                overall_best["C"] = C
                overall_best["model"] = "linsvc"
                overall_best["ngram_range"] = vc["ngram_range"]
                overall_best["max_features"] = vc["max_features"]

    print(
        "Best validation config: "
        f"model={overall_best['model']}, "
        f"ngrams={overall_best['ngram_range']}, "
        f"max_features={overall_best['max_features']}, "
        f"C={overall_best['C']} (acc={overall_best['acc']:.4f})"
    )

    # Retrain final vectorizer + model on all data with best config
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=overall_best["ngram_range"],
        max_features=overall_best["max_features"],
        min_df=3,
        max_df=0.8,
    )
    X = vectorizer.fit_transform(texts)

    if overall_best["model"] == "logreg":
        clf = LogisticRegression(
            C=overall_best["C"],
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            n_jobs=-1,
        )
    else:
        clf = LinearSVC(C=overall_best["C"])
    clf.fit(X, y)

    print("Generating category predictions with sklearn model...")
    test_reviews = list(readGz("test_Category.json.gz"))
    test_texts = [r["review_text"] for r in test_reviews]
    X_test = vectorizer.transform(test_texts)
    preds = clf.predict(X_test)

    with open("predictions_Category.csv", "w") as predictions:
        predictions.write("userID,reviewID,prediction\n")
        for r, p in zip(test_reviews, preds):
            predictions.write(f"{r['user_id']},{r['review_id']},{int(p)}\n")

    print("Category predictions complete! (sklearn TF-IDF + LogisticRegression)")

else:
    # ---------------------------------------------------------------------
    # Fallback: custom dictionary logistic regression with engineered features
    # ---------------------------------------------------------------------
    print("sklearn not available; using custom dictionary logistic regression.")

    def build_dictionary_from_reviews(reviews, vocab_size=4000):
        from collections import Counter

        cnt = Counter()
        for r in reviews:
            toks = tokenize(r["review_text"])
            cnt.update(toks)
        words = [w for w, _ in cnt.most_common(vocab_size)]
        wordId = {w: i for i, w in enumerate(words)}
        wordSet = set(words)
        return words, wordId, wordSet

    WORDS, WORDID, WORDSET = build_dictionary_from_reviews(train_reviews, vocab_size=4000)

    DIM = len(WORDS) + 4
    IDX_LEN = len(WORDS)
    IDX_UNIQ = len(WORDS) + 1
    IDX_AVGLEN = len(WORDS) + 2
    IDX_BIAS = len(WORDS) + 3

    def featurize_cat(tokens):
        """
        Build sparse features:
          - counts over top-vocab words, transformed with log1p
          - three extra features: normalized length, unique count, avg token length
          - L2-normalized, plus bias term.
        Returns a list of (index, value) pairs.
        """
        counts = defaultdict(float)
        n_tokens = len(tokens)
        uniq_tokens = len(set(tokens))
        if n_tokens:
            avg_len = sum(len(t) for t in tokens) / n_tokens
        else:
            avg_len = 0.0

        for w in tokens:
            if w in WORDSET:
                idx = WORDID[w]
                counts[idx] += 1.0

        # log1p for counts
        for idx in list(counts.keys()):
            counts[idx] = math.log1p(counts[idx])

        # Document-level features
        len_feat = min(1.0, n_tokens / 200.0)
        uniq_feat = min(1.0, uniq_tokens / 200.0) if n_tokens else 0.0
        avglen_feat = min(1.0, avg_len / 10.0) if n_tokens else 0.0

        counts[IDX_LEN] = len_feat
        counts[IDX_UNIQ] = uniq_feat
        counts[IDX_AVGLEN] = avglen_feat

        # L2 normalize
        norm_sq = sum(v * v for v in counts.values())
        if norm_sq > 0.0:
            norm = math.sqrt(norm_sq)
            for idx in list(counts.keys()):
                counts[idx] /= norm

        # Add intercept feature
        counts[IDX_BIAS] = 1.0

        return list(counts.items())

    def train_cat_logreg(reviews, lr_cat, reg_cat, epochs_cat):
        """Train multiclass logistic regression on given reviews, return weights."""
        weights = [[0.0] * DIM for _ in range(num_classes)]

        for epoch in range(epochs_cat):
            random.shuffle(reviews)
            correct = 0
            total = 0
            for review in reviews:
                y = review["genreID"]
                tokens = tokenize(review["review_text"])
                feats = featurize_cat(tokens)

                scores = [0.0] * num_classes
                for c in range(num_classes):
                    wc = weights[c]
                    s = 0.0
                    for j, v in feats:
                        s += wc[j] * v
                    scores[c] = s

                max_s = max(scores)
                exps = [math.exp(s - max_s) for s in scores]
                Z = sum(exps)
                probs = [e / Z for e in exps]

                pred_c = max(range(num_classes), key=lambda c: probs[c])
                if pred_c == y:
                    correct += 1
                total += 1

                for c in range(num_classes):
                    target = 1.0 if c == y else 0.0
                    diff = target - probs[c]
                    wc = weights[c]
                    for j, v in feats:
                        wc[j] += lr_cat * (diff * v - reg_cat * wc[j])

            acc = correct / total if total else 0.0
            print(f"  epoch {epoch + 1}/{epochs_cat} - train accuracy: {acc:.4f}")

        return weights

    def eval_cat_accuracy(reviews, weights):
        """Compute classification accuracy on reviews for given weights."""
        correct = 0
        total = 0
        for review in reviews:
            y = review["genreID"]
            tokens = tokenize(review["review_text"])
            feats = featurize_cat(tokens)

            scores = [0.0] * num_classes
            for c in range(num_classes):
                wc = weights[c]
                s = 0.0
                for j, v in feats:
                    s += wc[j] * v
                scores[c] = s

            pred_c = max(range(num_classes), key=lambda c: scores[c])
            if pred_c == y:
                correct += 1
            total += 1
        return (correct / total) if total else 0.0

    # Hyperparameter tuning
    indices = list(range(len(train_reviews)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_split = [train_reviews[i] for i in train_idx]
    val_split = [train_reviews[i] for i in val_idx]

    print(f"Tuning category model on {len(train_split)} train and {len(val_split)} validation examples...")

    config_grid = [
        {"lr": 0.2, "reg": 1e-5, "epochs": 6},
        {"lr": 0.1, "reg": 1e-5, "epochs": 8},
        {"lr": 0.1, "reg": 3e-5, "epochs": 8},
        {"lr": 0.05, "reg": 1e-5, "epochs": 10},
    ]

    best_cfg = None
    best_val_acc = -1.0

    for cfg in config_grid:
        print(f"Trying config: lr={cfg['lr']}, reg={cfg['reg']}, epochs={cfg['epochs']}")
        w_tmp = train_cat_logreg(train_split, cfg["lr"], cfg["reg"], cfg["epochs"])
        val_acc = eval_cat_accuracy(val_split, w_tmp)
        print(f"  -> validation accuracy: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_cfg = cfg

    print(
        f"Best config: lr={best_cfg['lr']}, reg={best_cfg['reg']}, "
        f"epochs={best_cfg['epochs']}, val_acc={best_val_acc:.4f}"
    )

    print("Training final category model on all training data with best hyperparameters...")
    weights = train_cat_logreg(train_reviews, best_cfg["lr"], best_cfg["reg"], best_cfg["epochs"])

    print("Generating category predictions (dictionary logistic regression with engineered features)...")

    with open("predictions_Category.csv", "w") as predictions:
        predictions.write("userID,reviewID,prediction\n")

        for review in readGz("test_Category.json.gz"):
            tokens = tokenize(review["review_text"])
            feats = featurize_cat(tokens)

            scores = [0.0] * num_classes
            for c in range(num_classes):
                wc = weights[c]
                s = 0.0
                for j, v in feats:
                    s += wc[j] * v
                scores[c] = s

            pred_c = max(range(num_classes), key=lambda c: scores[c])
            predictions.write(
                review["user_id"] + "," + review["review_id"] + "," + str(pred_c) + "\n"
            )

    print("Category predictions complete! (fallback dictionary model)")

print("\n" + "=" * 60)
print("All predictions generated successfully!")
print("Files created:")
print("  - predictions_Rating.csv")
print("  - predictions_Read.csv")
print("  - predictions_Category.csv")
print("=" * 60)

