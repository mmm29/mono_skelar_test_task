import pandas as pd
import numpy as np

from nameparser import HumanName
from gender_detection import GenderDetector
import gender_guesser.detector as gender
import string
from name_clusters import token_set_similarity, normalize_name, NameClusterer


CURRENCY_USD_EXCHANGE_RATES = {"USD": 1, "EUR": 0.87, "GBP": 0.75}

USER_FEATURES = [
    "card_max_nusers",
    "card_max_nuses",
    "ncard_holders",
    "no_cardholder_rate",
    "invalid_cardholder_rate",
    "ncards",
    "ncard_country",
    "npayment_country",
    "tx_cnt",
    "tx_rate",
    "total_tx_time",
    "tx_time_since_reg",
    "country_mismatch_rate",
    "total_amount",
    "mean_amount",
    "ntx_type",
    "min_amount",
    "max_amount",
    "no_card_country_rate",
    "no_payment_country_rate",
    "card_holder_rate_male_gender",
    "card_holder_rate_female_gender",
    "card_holder_rate_unknown_gender",
    "fail_rate",
    "fail_cnt",
    "ncurrency",
    "reg_payment_country_mismatch_rate",
    "reg_card_country_mismatch_rate",
    "max_n_parts_cardholder",
    "mean_n_parts_cardholder",
    "name_paired_rate",
    "email_username_ndigits",
    "email_username_length",
    "email_username_digit_ratio",
    "email_username_has_uppercase",
    "email_domain",
    "is_valid_email",
    "no_email",
    "email_nusers",
    "reg_time_hour",
    "reg_time_dayofweek",
    "n_cardholder_clusters",
    "reg_country",
    "traffic_type",
    # "mean_card_mean_fraud",
    # "max_card_mean_fraud",
    # "mean_card_sum_fraud",
    # "max_card_sum_fraud",
    "median_tx_time_gap",
    "min_tx_time_gap",
    "first_tx_error_group",
    "initial_fails",
    "email_contains_first_name_rate",
    "email_contains_last_name_rate",
    "email_contains_card_holder_word_rate",
    "success_cnt",
    "email_card_holder_similarity_rate",
    "card_holder_n_names_rate",
    "card_holder_n_names_max",
    "card_reg_countries_mean",
    "card_reg_countries_max",
    "card_payment_countries_mean",
    "card_payment_countries_max",
]

TX_FEATURES = [
    "name_paired",
    "is_male",
    "no_cardholder",
    "invalid_cardholder",
    "country_mismatch",
    "reg_card_country_mismatch",
    "reg_payment_country_mismatch",
    "amount_usd",
    "n_parts_cardholder",
    "card_nusers",
    "card_nuses",
    "payment_country",
    "card_country",
    "card_type",
    "card_brand",
    "currency",
    "error_group",
    "transaction_type",
    "card_holder_gender",
    "is_fail",
    "reg_country",
    "traffic_type",
    "email_contains_first_name",
    "email_contains_last_name",
    "email_contains_card_holder_word",
    "email_card_holder_similarity",
    "card_holder_n_names",
    "payment_country_changed",
    "card_country_changed",
    "amount_change",
    "card_holder_change_similarity",
    "card_reg_countries",
    "card_payment_countries",
    # "card_mean_fraud",
    # "card_sum_fraud",
]


class EngineeredFeatures:
    def __init__(
        self,
        users: pd.DataFrame,
        tx: pd.DataFrame,
        users_features: list[str],
        tx_features: list[str],
    ) -> None:
        self.users = users
        self.tx = tx
        self.user_features = users_features
        self.tx_features = tx_features


def engineer_features(tx: pd.DataFrame, users: pd.DataFrame) -> EngineeredFeatures:
    tx = tx.copy()
    users = users.copy()

    users["is_male"] = users["gender"] == "male"
    COPY_COLS = ["email", "reg_country", "traffic_type", "is_male", "is_fraud"]
    tx[COPY_COLS] = users.loc[tx["id_user"], COPY_COLS].reset_index(drop=True)

    exchange_rates = tx["currency"].map(CURRENCY_USD_EXCHANGE_RATES)
    tx["amount_usd"] = tx["amount"] / exchange_rates

    tx["timestamp_tr_dt"] = pd.to_datetime(tx["timestamp_tr"], format="mixed")
    users["timestamp_reg_dt"] = pd.to_datetime(users["timestamp_reg"], format="mixed")

    engineer_card_holder_features(tx)
    engineer_card_holder_email_features(tx)
    engineer_card_features(tx, users)
    engineer_prev_tx_features(tx)
    engineer_tx_features(tx)
    users = engineer_user_tx_features(users, tx)
    users, users_cat_features = engineer_users_category_features(users, tx)

    return EngineeredFeatures(
        users, tx, USER_FEATURES + users_cat_features, TX_FEATURES
    )


def engineer_card_holder_features(tx: pd.DataFrame):
    gender_detector = GenderDetector()
    unique_names = tx["card_holder"].dropna().str.lower().drop_duplicates()
    name_genders = unique_names.map(gender_detector.detect_gender)
    tx["card_holder_gender"] = (
        tx["card_holder"]
        .str.lower()
        .map(pd.Series(name_genders.values, index=unique_names.values))
    )

    ch = tx["card_holder"].fillna("")
    parts = ch.str.split()
    tx["n_parts_cardholder"] = parts.str.len()
    tx["name_paired"] = tx["n_parts_cardholder"].eq(2) & parts.str[0].eq(parts.str[1])
    tx["no_cardholder"] = tx["card_holder"].isna()
    tx["invalid_cardholder"] = ~ch.str.contains(" ")
    tx["card_holder_normalized"] = ch.str.lower()

    gd = gender.Detector(case_sensitive=False)

    def card_holder_n_names(ch: str) -> int:
        s = ch.split()
        return sum(gd.get_gender(c) != "unknown" for c in s)

    tx["card_holder_n_names"] = tx["card_holder_normalized"].apply(card_holder_n_names)

    def detect_names(name: str | None):
        if not isinstance(name, str):
            return "", ""
        parsed = HumanName(name)
        return parsed.first, parsed.last

    parsed_names = [(n, *detect_names(n)) for n in unique_names]
    parsed_names = pd.DataFrame(
        parsed_names, columns=["name", "first_name", "last_name"]
    ).set_index("name")

    tx["card_holder_first_name"] = (
        tx["card_holder_normalized"]
        .map(parsed_names["first_name"])
        .str.lower()
        .fillna("")
    )
    tx["card_holder_last_name"] = (
        tx["card_holder_normalized"]
        .map(parsed_names["last_name"])
        .str.lower()
        .fillna("")
    )


def engineer_card_holder_email_features(tx: pd.DataFrame):
    tx["email_lower"] = tx["email"].str.lower().fillna("")
    tx["email_contains_first_name"] = tx.apply(
        lambda row: (
            len(row["card_holder_first_name"]) > 1
            and row["card_holder_first_name"] in row["email_lower"]
        ),
        axis=1,
    )
    tx["email_contains_last_name"] = tx.apply(
        lambda row: (
            len(row["card_holder_last_name"]) > 1
            and row["card_holder_last_name"] in row["email_lower"]
        ),
        axis=1,
    )
    tx["email_contains_card_holder_word_n"] = tx.apply(
        lambda row: sum(
            [
                w in row["email_lower"]
                for w in row["card_holder_normalized"].split()
                if len(w) > 2
            ]
        ),
        axis=1,
    )
    tx["email_contains_card_holder_word"] = tx["email_contains_card_holder_word_n"] > 0

    def email_card_holder_similarity(row) -> float | None:
        email, cardholder = row["email_lower"], row["card_holder_normalized"]

        if len(email) == 0 or len(cardholder) == 0:
            return None

        at_idx = email.find("@")
        assert at_idx != -1
        email_username = email[:at_idx]

        def retain_allowed_chars(s):
            return "".join(c for c in s if c in string.ascii_lowercase)

        email_username = retain_allowed_chars(email_username)
        normalized_name = normalize_name(cardholder)

        return token_set_similarity(email_username, normalized_name)

    tx["email_card_holder_similarity"] = tx.apply(email_card_holder_similarity, axis=1)


def engineer_card_features(tx: pd.DataFrame, users: pd.DataFrame):
    card_grp = tx.groupby("card_mask_hash")

    card_nusers = card_grp["id_user"].nunique()
    card_nuses = card_grp["id_user"].size()

    tr = tx[["card_mask_hash", "id_user", "payment_country"]].drop_duplicates(
        ["card_mask_hash", "id_user", "payment_country"]
    )
    tr = tr.merge(
        users[["reg_country"]],
        on="id_user",
        how="left",
    )
    card_stats = tr.groupby("card_mask_hash", sort=False).agg(
        card_reg_countries=("reg_country", "nunique"),
        card_payment_countries=("payment_country", "nunique"),
    )

    tx["card_nusers"] = tx["card_mask_hash"].map(card_nusers)
    tx["card_nuses"] = tx["card_mask_hash"].map(card_nuses)
    tx["card_reg_countries"] = tx["card_mask_hash"].map(
        card_stats["card_reg_countries"]
    )
    tx["card_payment_countries"] = tx["card_mask_hash"].map(
        card_stats["card_payment_countries"]
    )


def engineer_prev_tx_features(tx: pd.DataFrame):
    tr = tx.sort_values(["id_user", "timestamp_tr_dt"])
    tr["prev_tx_id"] = tr.index.to_series().groupby(tr["id_user"]).shift()
    tx["prev_tx_id"] = tr["prev_tx_id"].reindex(tx.index)

    tx["payment_country_changed"] = (
        tx["prev_tx_id"].map(tx["payment_country"]) != tx["payment_country"]
    )
    tx["card_country_changed"] = (
        tx["prev_tx_id"].map(tx["card_country"]) != tx["card_country"]
    )
    tx["amount_change"] = tx["amount_usd"] - tx["prev_tx_id"].map(tx["amount_usd"])

    def card_holder_change_similarity(row):
        if not isinstance(row["prev_tx_card_holder"], str):
            return None

        a, b = row["prev_tx_card_holder"], row["card_holder_normalized"]
        return token_set_similarity(normalize_name(a), normalize_name(b))

    tx["prev_tx_card_holder"] = tx["prev_tx_id"].map(tx["card_holder_normalized"])
    tx["card_holder_change_similarity"] = tx.apply(
        card_holder_change_similarity, axis=1
    )

    tx["tx_time_gap"] = (
        tx.sort_values(["id_user", "timestamp_tr_dt"])
        .groupby("id_user")["timestamp_tr_dt"]
        .diff()
    )


def engineer_tx_features(tx: pd.DataFrame):
    tx["is_fail"] = tx["status"].eq("fail")
    tx["is_success"] = tx["status"].eq("success")
    tx["no_card_country"] = tx["card_country"].isna()
    tx["no_payment_country"] = tx["payment_country"].isna()
    tx["country_mismatch"] = tx["card_country"] != tx["payment_country"]
    tx["reg_card_country_mismatch"] = tx["card_country"] != tx["reg_country"]
    tx["reg_payment_country_mismatch"] = tx["payment_country"] != tx["reg_country"]


def engineer_user_tx_features(users: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    users["median_tx_time_gap"] = (
        tx.groupby("id_user")["tx_time_gap"].median().dt.total_seconds() / 3600
    )
    users["min_tx_time_gap"] = (
        tx.groupby("id_user")["tx_time_gap"].min().dt.total_seconds() / 3600
    )
    users["first_tx_error_group"] = tx.groupby("id_user")["error_group"].first()

    tr = tx.sort_values(["id_user", "timestamp_tr"])
    is_success = tr["status"].eq("success")
    success_cum = is_success.groupby(tr["id_user"]).cumsum()
    before_first_success = success_cum.eq(0)
    initial_fails = (
        (before_first_success & tr["status"].eq("fail"))
        .groupby(tr["id_user"])
        .sum()
        .astype("int64")
    )
    users["initial_fails"] = initial_fails

    user_tx_agg = tx.groupby("id_user", sort=False).agg(
        first_tx_time=("timestamp_tr_dt", "min"),
        last_tx_time=("timestamp_tr_dt", "max"),
        tx_cnt=("id_user", "size"),
        max_n_parts_cardholder=("n_parts_cardholder", "max"),
        mean_n_parts_cardholder=("n_parts_cardholder", "mean"),
        name_paired_rate=("name_paired", "mean"),
        ncurrency=("currency", "nunique"),
        ncard_holders=("card_holder_normalized", "nunique"),
        ncards=("card_mask_hash", "nunique"),
        ncard_country=("card_country", "nunique"),
        ntx_type=("transaction_type", "nunique"),
        npayment_country=("payment_country", "nunique"),
        country_mismatch_rate=("country_mismatch", "mean"),
        total_amount=("amount_usd", "sum"),
        min_amount=("amount_usd", "min"),
        max_amount=("amount_usd", "max"),
        mean_amount=("amount_usd", "mean"),
        std_amount=("amount_usd", "std"),
        no_cardholder_rate=("no_cardholder", "mean"),
        invalid_cardholder_rate=("invalid_cardholder", "mean"),
        fail_rate=("is_fail", "mean"),
        fail_cnt=("is_fail", "sum"),
        success_cnt=("is_success", "sum"),
        no_card_country_rate=("no_card_country", "mean"),
        no_payment_country_rate=("no_payment_country", "mean"),
        card_max_nusers=("card_nusers", "max"),
        card_max_nuses=("card_nuses", "max"),
        # mean_card_mean_fraud=("card_mean_fraud", "mean"),
        # max_card_mean_fraud=("card_mean_fraud", "max"),
        # mean_card_sum_fraud=("card_sum_fraud", "mean"),
        # max_card_sum_fraud=("card_sum_fraud", "max"),
        email_contains_first_name_rate=("email_contains_first_name", "mean"),
        email_contains_last_name_rate=("email_contains_last_name", "mean"),
        email_contains_card_holder_word_rate=(
            "email_contains_card_holder_word",
            "mean",
        ),
        email_card_holder_similarity_rate=("email_card_holder_similarity", "mean"),
        card_holder_n_names_rate=("card_holder_n_names", "mean"),
        card_holder_n_names_max=("card_holder_n_names", "max"),
        card_holder_change_similarity_mean=("card_holder_change_similarity", "mean"),
        card_reg_countries_mean=("card_reg_countries", "mean"),
        card_reg_countries_max=("card_reg_countries", "max"),
        card_payment_countries_mean=("card_payment_countries", "mean"),
        card_payment_countries_max=("card_payment_countries", "max"),
        reg_payment_country_mismatch_rate=("reg_payment_country_mismatch", "mean"),
        reg_card_country_mismatch_rate=("reg_card_country_mismatch", "mean"),
    )

    users = users.drop(columns=users.columns.intersection(user_tx_agg.columns))
    users = users.join(user_tx_agg, on="id_user")

    users["card_holder_has_male_gender"] = (
        tx.assign(d=tx["card_holder_gender"] == "male").groupby("id_user")["d"].any()
    )
    users["card_holder_has_female_gender"] = (
        tx.assign(d=tx["card_holder_gender"] == "female").groupby("id_user")["d"].any()
    )
    users["card_holder_rate_male_gender"] = (
        tx.assign(d=tx["card_holder_gender"] == "male").groupby("id_user")["d"].mean()
    )
    users["card_holder_rate_female_gender"] = (
        tx.assign(d=tx["card_holder_gender"] == "female").groupby("id_user")["d"].mean()
    )
    users["card_holder_rate_unknown_gender"] = (
        tx.assign(d=tx["card_holder_gender"] == "unknown")
        .groupby("id_user")["d"]
        .mean()
    )
    users["gender_card_holder_mismatch"] = (
        users["is_male"] & users["card_holder_has_female_gender"]
    ) | (~users["is_male"] & users["card_holder_has_male_gender"])

    name_clusterer = NameClusterer()
    users["n_cardholder_clusters"] = tx.groupby("id_user", sort=False)[
        "card_holder"
    ].apply(lambda x: name_clusterer.num_clusters(x.dropna().tolist()))

    # Email features
    users["email_username"] = users["email"].str.split("@").str[0]
    users["email_username_ndigits"] = users["email_username"].str.count(r"\d")
    users["email_username_length"] = users["email_username"].str.len()
    users["email_username_digit_ratio"] = (
        users["email_username_ndigits"] / users["email_username_length"]
    )
    users["email_username_has_uppercase"] = users["email_username"].str.contains(
        r"[A-Z]", na=False
    )
    users["email_domain"] = users["email"].str.split("@").str[1]

    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    users["no_email"] = users["email"].isna()
    users["is_valid_email"] = users["email"].str.fullmatch(pattern, na=True)
    email_counts = users["email"].value_counts()
    users["email_nusers"] = users["email"].map(email_counts)

    users["reg_time_hour"] = users["timestamp_reg_dt"].dt.hour
    users["reg_time_dayofweek"] = users["timestamp_reg_dt"].dt.dayofweek

    users["total_tx_time"] = (
        (users["last_tx_time"] - users["first_tx_time"]).dt.total_seconds() / 3600 / 24
    )
    users["tx_rate"] = users["tx_cnt"] / users["total_tx_time"]
    users["tx_rate"] = users["tx_rate"].replace(np.inf, 0)
    users["tx_time_since_reg"] = (
        users["first_tx_time"] - users["timestamp_reg_dt"]
    ).dt.total_seconds() / 3600

    return users


def engineer_users_category_features(
    users: pd.DataFrame, tx: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    features = []

    CATEGORIES = ["card_type", "transaction_type", "error_group"]

    for cat in CATEGORIES:
        d = pd.get_dummies(tx[cat], prefix=cat, dummy_na=False)
        cnt = d.groupby(tx["id_user"]).sum().add_prefix("cnt_")
        rate = d.groupby(tx["id_user"]).mean().add_prefix("rate_")
        users = users.drop(
            columns=users.columns.intersection(
                cnt.columns.tolist() + rate.columns.tolist()
            )
        )
        users = users.join(cnt, on="id_user").join(rate, on="id_user")
        features += cnt.columns.tolist()
        features += rate.columns.tolist()

    return users, features
