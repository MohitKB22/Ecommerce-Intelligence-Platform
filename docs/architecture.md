# Architecture

## 1. System architecture

```mermaid
flowchart TD
    subgraph Client
        UI["React + TypeScript SPA<br/>11 routes, TanStack Query"]
    end

    subgraph Edge
        NGX["Nginx<br/>static assets + /api proxy"]
    end

    subgraph API["FastAPI application"]
        MW["Middleware<br/>request id · rate limit · security headers · gzip"]
        R1["Products"]; R2["Search"]; R3["Recommendations"]
        R4["Events"]; R5["Intelligence"]; R6["Analytics + ML (admin)"]
    end

    subgraph Services
        PS["Product service"]; SS["Search service"]; RS["Recommendation service"]
        PZ["Personalization"]; AS["Analytics"]; MS["Monitoring"]
    end

    subgraph MLLayer["ML prediction layer"]
        Store["Model store<br/>lazy load · LRU cache · hot reload · latency capture"]
        Art["Artifacts<br/>Recommender · Sentiment · Forecast · Price · Segmentation"]
    end

    subgraph Data
        FS["Feature store<br/>versioned + timestamped"]
        PG[("PostgreSQL / SQLite<br/>17 tables")]
        RD[("Redis / in-process cache")]
        VS[("Vector index<br/>numpy or pgvector")]
    end

    subgraph Offline
        Gen["Synthetic data generator"]
        Train["Training pipelines x5"]
        Eval["Evaluation + baselines"]
        Reg[("Model registry")]
        WK["Background worker"]
    end

    UI --> NGX --> MW --> R1 & R2 & R3 & R4 & R5 & R6
    R1 --> PS; R2 --> SS; R3 --> RS
    R5 --> MLLayer; R6 --> AS & MS
    PS & SS & RS --> PZ
    RS & SS --> Store --> Art
    PS & SS & RS & PZ & AS --> FS
    FS --> PG & RD
    SS --> VS
    PG --> Gen --> Train --> Eval --> Reg --> Store
    WK --> FS & Store & PG
```

## 2. Recommendation pipeline

```mermaid
flowchart LR
    subgraph Signals
        E["User events<br/>view · click · cart · purchase"]
        O["Order history"]
        C["Product content<br/>title · description · specs"]
    end

    E --> IM["Implicit matrix<br/>confidence-weighted"]
    IM --> BM["BM25 re-weighting<br/>removes popularity bias"]
    BM --> SVD["Truncated SVD<br/>32 latent factors"]
    SVD --> UF["User factors"] & IF["Item factors"]
    IM --> KNN["Item-item cosine<br/>top-50 neighbours"]
    C --> EMB["TF-IDF + SVD embeddings"]
    O --> CO["Co-purchase graph<br/>lift-weighted"]

    UF & IF & KNN --> COL["Collaborative score"]
    EMB --> CON["Content score"]
    E --> POP["Popularity + trending"]

    COL --> RES1["Residualise vs popularity"]
    CON --> RES2["Residualise vs popularity"]

    RES1 & RES2 & POP & PER["Personalization<br/>category/brand affinity"] & BIZ["Business<br/>margin · conversion · stock"] --> BLEND["Weighted blend<br/>weights fitted on validation"]
    BLEND --> RANK["Rank · exclude history · damp out-of-stock"]
    RANK --> OUT["Top-K + per-signal explanation"]
    CO --> FBT["Frequently bought together"]
```

**Cold start.** Unknown users, or users with fewer than three interactions, are
routed to a popularity + trending + rating blend, optionally scoped to their
strongest category affinity. New products are reachable through content
similarity because their embedding exists from the moment they are indexed.

## 3. Search pipeline

```mermaid
flowchart TD
    Q["Query"] --> N["Normalise"]
    N --> T["Typo correction<br/>bounded Damerau-Levenshtein vs vocabulary"]
    T --> I["Intent parsing<br/>budget · cheap/premium · in-stock"]
    I --> P{"Empty query?"}
    P -->|yes| BR["Browse mode<br/>filter + popularity"]
    P -->|no| RET["Retrieval"]

    RET --> BM25["BM25 lexical<br/>top-300"]
    RET --> SEM["Dense vector search<br/>top-300"]
    BM25 & SEM --> CAND["Candidate union"]
    CAND --> FLT["Apply filters<br/>category · brand · price · rating · stock"]
    FLT --> Z{"Empty?"}
    Z -->|yes| FB["Semantic-only fallback<br/>over the whole catalogue"]
    Z -->|no| SC["Score 7 signals"]
    FB --> SC

    SC --> RK["Weighted blend<br/>configurable weights"]
    RK --> SORT["Sort mode"] --> PAGE["Paginate + facets"]
    PAGE --> LOG["Log SearchEvent<br/>CTR / abandonment / latency"]
```

Ranking signals: text relevance · semantic similarity · popularity · rating ·
conversion probability · user preference · availability. Each result returns its
signal vector and a human-readable explanation.

## 4. ML training pipeline

```mermaid
flowchart LR
    DB[("Operational database")] --> L["Load<br/>ml/preprocessing/loaders.py"]
    L --> V["Validate<br/>row counts · ranges · duplicates"]
    V -->|fails| STOP["DataValidationError<br/>exit 2"]
    V -->|passes| PRE["Preprocess<br/>temporal split · feature engineering"]
    PRE --> TR["Train"]
    TR --> EV["Evaluate vs baseline"]
    EV --> SV["Save joblib artifact"]
    SV --> MT["Write metrics JSON"]
    MT --> RG["Register in manifest"]
    RG --> DBW["Mirror to model_registry table"]
    RG --> SIDE["Side effects<br/>segments · forecasts · price rows"]
```

Every pipeline subclasses `TrainingPipeline` and implements only
`preprocess / train / evaluate`; the template method owns validation, timing,
persistence and registration.

## 5. Demand forecasting pipeline

```mermaid
flowchart TD
    OI["order_items + orders"] --> DD["Daily demand per SKU"]
    DD --> WK["Aggregate to ISO weeks<br/>(daily is mostly zeros)"]
    WK --> ELIG["Filter: >=25 units and >=20 weeks"]
    ELIG --> PANEL["Balanced panel<br/>gaps filled with zero"]
    PANEL --> FE["Features<br/>lags 1-12w · rolling mean/std · calendar<br/>cyclical encodings · price ratio · promo"]
    FE --> SPLIT["Temporal split<br/>last 8 weeks held out"]
    SPLIT --> BASE["Seasonal naive<br/>recent 4w level x month factor"]
    BASE --> RESID["Target = units - baseline"]
    RESID --> GBM["HistGradientBoostingRegressor"]
    GBM --> CMP{"Beats naive on MAE?"}
    CMP -->|yes| SEL["naive + GBM residual"]
    CMP -->|no| NAI["seasonal naive"]
    SEL & NAI --> CI["Residual spread -> 95% interval<br/>widens with sqrt(horizon)"]
    CI --> SERVE["Recursive weekly rollout<br/>-> daily points -> 7/14/30d totals"]
```

## 6. Customer segmentation

```mermaid
flowchart LR
    O["Orders"] & E["Events"] & R["Reviews"] --> F["RFM + behavioural features<br/>recency · frequency · monetary · AOV<br/>diversity · sessions · discount affinity · tenure"]
    F --> LOG["log1p on heavy-tailed columns"]
    LOG --> SC["StandardScaler"]
    SC --> K["K-Means for k in 4..8"]
    K --> SIL["Silhouette per k"]
    SIL --> PICK["Largest k within 88% of best<br/>(silhouette alone is too coarse)"]
    PICK --> NAME["Greedy centroid -> segment name<br/>each name used at most once"]
    NAME --> PERSIST["Write customer_segments"]
```

## 7. Data flow

```mermaid
sequenceDiagram
    participant U as Shopper
    participant F as Frontend
    participant A as API
    participant C as Cache
    participant M as Model store
    participant D as Database

    U->>F: Open homepage
    F->>A: GET /recommendations/homepage
    A->>C: profile cache lookup
    alt miss
        A->>D: events, orders, reviews
        A->>C: store profile (TTL 300s)
    end
    A->>M: recommender artifact
    M-->>A: scores + components
    A->>D: log impressions
    A-->>F: sections + explanations

    U->>F: Click a product
    F->>A: POST /events {click}
    A->>D: persist event
    A->>D: attribute click to impression
    A->>C: invalidate profile
    Note over D: the event becomes training signal
```

## 8. Deployment architecture

```mermaid
flowchart TD
    subgraph Host["docker compose"]
        FE["frontend<br/>nginx :80 -> :3000"]
        BE["backend<br/>uvicorn :8000"]
        WK["worker<br/>scheduler"]
        MG["migrate<br/>one-shot: migrate + seed + train"]
        PG[("postgres:16")]
        RD[("redis:7")]
        VD[("pgvector<br/>profile: vector")]
        GW["gateway<br/>profile: gateway"]
    end

    FE -->|/api proxy| BE
    BE --> PG & RD
    WK --> PG & RD
    MG --> PG
    BE -.optional.-> VD
    GW --> FE & BE
    MG -->|completes first| BE

    VOL1[("model_data volume")] --- BE & WK & MG
    VOL2[("seed_data volume")] --- BE & WK & MG
```

Health checks are defined on every container; `backend` waits for `postgres`,
`redis` and successful completion of `migrate` before accepting traffic.

## Design decisions

**Why residualise recommendation components?** Measured: the naive weighted blend
scored NDCG@10 0.036 against a popularity baseline of 0.085 — 58% *worse*. Raw
collaborative and content scores are dominated by the popularity direction, so
adding them to a blend that already scores popularity just adds noise. Projecting
that direction out makes each component contribute orthogonal information, and
the same blend then beats the baseline.

**Why fit blend weights instead of hard-coding them?** The specified starting
weights (0.35/0.25/0.15/0.15/0.10) were tuned for a different data distribution.
Fitting on a validation slice of the post-cutoff window — never the test slice —
moved held-out NDCG@10 from 0.037 to 0.062. The fitted weights are stored in the
artifact and remain overridable per request.

**Why an in-process vector index by default?** For a catalogue of this size an
exact cosine scan is a single dense matmul: sub-millisecond, no approximation
error, no extra stateful service. pgvector is wired and ready behind
`VECTOR_DB_URL` for catalogues where that stops being true.

**Why weekly forecasting?** The median SKU sells fewer than one unit per day, so
a daily target is mostly zeros and the model degenerates to predicting zero.
Weekly aggregation is standard retail practice and preserves the seasonality that
actually drives replenishment.
