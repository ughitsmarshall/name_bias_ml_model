
## Evaluating Name Bias in Election Results: Machine Learning Project
### Marshall Carey-Matthews, Nov. 2026
""" Data Needed:
- "1976-2024-house.csv": contains data on US House Elections (1976-2024)
- "1976-2024-senate.csv": contains data on US Senate Elections (1976-2024)
- "Names2020_FirstNames_RaceHispanic.csv": Census Data on Name-Race Frequency
- "Names2020_FirstNames_Sex.csv": Census Data on Name-Sex Frequency
- "Names2020_LastNames_RaceHispanic.csv": Census Data on Name-Race Frequency
"""

import re
import pandas as pd
import numpy as np
import networkx as nx

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.dummy import DummyRegressor

"""Step 1: Reconciling Elections Data"""

# Grab the CSV files and read them into dictionaries
house_data = pd.read_csv('1976-2024-house.csv', low_memory=False)
senate_data = pd.read_csv('1976-2024-senate.csv', low_memory=False)

# Standardize casing and strip whitespace
for file in (house_data, senate_data):
    for col in ['stage', 'mode', 'state', 'state_po', 'office', 'district', 'candidate']:
        if col in file.columns:
            file[col] = file[col].astype("string").str.strip().str.upper()

# Create "party_detailed" column for House data and shift "party" into that category,
house_data = house_data.rename(columns={'party': 'party_detailed'})
# clean up the data,
house_data['party_detailed'] = house_data['party_detailed'].astype('string').str.strip().str.upper()
senate_data['party_detailed'] = senate_data['party_detailed'].astype('string').str.strip().str.upper()
senate_data['party_simplified'] = senate_data['party_simplified'].astype('string').str.strip().str.upper()
# and convert verbose parties to simplified parties
dem_affiliates = {
    'DEMOCRAT', 'DEMOCRATIC', 'DEMOCRATIC-FARMER-LABOR',
    'DEMOCRATIC-FARM-LABOR', 'DEMOCRATIC-NPL', 'DEMOCRATIC-NONPARTISAN LEAGUE'
}
rep_affiliates = {'REPUBLICAN', 'INDEPENDENT-REPUBLICAN', 'GOP'}
  # Function to convert verbose to simple
def map_party(p):
    if pd.isna(p):
        return pd.NA
    if p in dem_affiliates:
        return 'DEMOCRAT'
    if p in rep_affiliates:
        return 'REPUBLICAN'
    if p == 'LIBERTARIAN':
        return 'LIBERTARIAN'
    return 'OTHER'
# Mapped to the new party_simplified category for house data
house_data['party_simplified'] = house_data['party_detailed'].apply(map_party)

# Make sure districts for House and Senate are both strings
house_data['district'] = house_data['district'].astype(str)
senate_data['district'] = senate_data['district'].astype(str)

# Convert all vote counts into integers
senate_data['candidatevotes'] = senate_data['candidatevotes'].round().astype('Int64')
senate_data['totalvotes'] = senate_data['totalvotes'].round().astype('Int64')
house_data['candidatevotes'] = house_data['candidatevotes'].astype('Int64')
house_data['totalvotes'] = house_data['totalvotes'].astype('Int64')

# Address lack of fusion ticket delineation in senate data
senate_data['runoff'] = senate_data['stage'].str.contains('RUNOFF', na=False)
senate_data['fusion_ticket'] = False

# Now combine house and senate data
elections = pd.concat([house_data, senate_data], ignore_index=True)
# Take out primaries and runoffs
gen = elections[
    (elections['stage'] == 'GEN') &
    (elections['runoff'] != True) &
    (elections['writein'] == False) &
    (elections['candidate'].notna()) &
    (~elections['candidate'].isin(['WRITEIN', 'WRITE-IN', 'UNDERVOTES', 'OVERVOTES', 'VOID',
    'BLANK', 'BLANKS', 'SCATTER', 'SCATTERING', 'OTHER', 'OTHERS',
    'MISCELLANEOUS', 'SPOILED', 'NONE OF THESE CANDIDATES'])) &
    (elections['candidatevotes'] > 0)
].copy()
# Map the fusion tickets to a single party
def resolve_candidate_party(parties):
    s = set(parties.dropna())
    if 'DEMOCRAT' in s and 'REPUBLICAN' not in s:
        return 'DEMOCRAT'
    if 'REPUBLICAN' in s and 'DEMOCRAT' not in s:
        return 'REPUBLICAN'
    if 'LIBERTARIAN' in s:
        return 'LIBERTARIAN'
    return 'OTHER'
# Then group all the candidates of a specific election together
candidates_clean = (
    gen.groupby(['year', 'state', 'state_po', 'office', 'district', 'special', 'candidate'], as_index=False)
    .agg(
        party_simplified=('party_simplified', resolve_candidate_party),
        candidatevotes=('candidatevotes', 'sum'),
        totalvotes=('totalvotes', 'first')
    )
)

# Sanity check for data
print("Number of election candidate rows:", len(candidates_clean))
print(candidates_clean['party_simplified'].value_counts())




"""Step 2: Pre-processing Census Names Data"""

# Load and clean the 2020 Census name tables
def load_census_table(filepath, name_col):
    df = pd.read_csv(filepath, skiprows=2, encoding='latin1', low_memory=False)
    df = df[df[name_col].notna() & (df[name_col] != 'ALL OTHER NAMES')].copy()
    df[name_col] = df[name_col].astype('string').str.strip().str.upper()
    for col in df.columns:
        if col not in {name_col, 'RANK'}:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    return df

first_race = load_census_table('Names2020_FirstNames_RaceHispanic.csv', 'FIRST NAME')
first_sex = load_census_table('Names2020_FirstNames_Sex.csv', 'FIRST NAME')
last_race = load_census_table('Names2020_LastNames_RaceHispanic.csv', 'LAST NAME')

first_race['first_pct_white'] = first_race['NON-HISPANIC OR LATINO WHITE ALONE'] / first_race['FREQUENCY (COUNT)']
first_sex['first_pct_male'] = first_sex['MALE'] / first_sex['FREQUENCY (COUNT)']
last_race['last_pct_white'] = last_race['NON-HISPANIC OR LATINO WHITE ALONE'] / last_race['FREQUENCY (COUNT)']

first_lookup = pd.merge(
    first_race[['FIRST NAME', 'first_pct_white']],
    first_sex[['FIRST NAME', 'first_pct_male']],
    on='FIRST NAME',
    how='outer'
)
last_lookup = last_race[['LAST NAME', 'last_pct_white']].copy()

# Parse ballot candidate strings into clean FIRST and LAST names

# Nickname map originally included by Gemini, removed for the following reasons:
  # 1: The way candidates advertise themselves change the way they are perceived
  # 2: Most of the common nicknames are included in the Census dataset anyhow
"""NICKNAME_MAP = {
    'BILL': 'WILLIAM', 'BOB': 'ROBERT', 'JIM': 'JAMES', 'DICK': 'RICHARD',
    'TOM': 'THOMAS', 'MIKE': 'MICHAEL', 'DAVE': 'DAVID', 'DAN': 'DANIEL',
    'CHUCK': 'CHARLES', 'KEN': 'KENNETH', 'RON': 'RONALD', 'RICK': 'RICHARD',
    'DON': 'DONALD', 'CHRIS': 'CHRISTOPHER', 'STEVE': 'STEVEN', 'PETE': 'PETER',
    'TONY': 'ANTHONY', 'JOE': 'JOSEPH', 'ED': 'EDWARD', 'FRED': 'FREDERICK',
    'GREG': 'GREGORY', 'JEFF': 'JEFFREY', 'TIM': 'TIMOTHY', 'NICK': 'NICHOLAS',
    'PAT': 'PATRICK', 'SAM': 'SAMUEL', 'BEN': 'BENJAMIN', 'CHARLIE': 'CHARLES'
}"""

first_set = set(first_lookup['FIRST NAME'].dropna())
last_set = set(last_lookup['LAST NAME'].dropna())

SUFFIXES = {'JR', 'SR', 'II', 'III', 'IV', 'V', 'MD', 'PHD', 'ESQ'}
PREFIXES = {'DR', 'REV', 'SEN', 'REP', 'GOV', 'GEN', 'COL', 'MAJ', 'CAPT', 'LT', 'SGT', 'HON', 'JUDGE', 'MR', 'MRS', 'MS'}
PARTICLES = {'DE', 'LA', 'LAS', 'LOS', 'DEL', 'VAN', 'VON', 'ST', 'SAN', 'SANTA', 'MC', 'MAC', 'DA', 'DI', 'DU', 'LE'}

def clean_alpha(tok):
    return re.sub(r'[^A-Z]', '', tok)

def parse_first_last(name):
    if pd.isna(name):
        return pd.Series([pd.NA, pd.NA])
    s = str(name).upper().replace('\\"', '"').replace("\\'", "'")

    # Capture quoted/parenthetical nicknames in case candidate only lists initials
    nicknames = re.findall(r'["\(]([^"\)]+)["\)]', s)
    nick_tokens = [clean_alpha(t) for n in nicknames for t in n.split() if len(clean_alpha(t)) > 1]

    # Remove quoted/parenthetical text (excluding apostrophes from delimiter class)
    s = re.sub(r'["\(][^"\)]*["\)]', ' ', s)
    s = re.sub(r'[.,]', ' ', s)
    tokens = [t for t in s.split() if t]

    while tokens and clean_alpha(tokens[0]) in PREFIXES:
        tokens.pop(0)
    while tokens and clean_alpha(tokens[-1]) in SUFFIXES:
        tokens.pop()
    if not tokens:
        return pd.Series([pd.NA, pd.NA])

    # Handle multi-word particles (e.g., DE LA GARZA, VAN HOLLEN) and hyphenated surnames
    raw_last = tokens[-1]
    last_clean = clean_alpha(raw_last)
    if len(tokens) >= 4 and clean_alpha(tokens[-3]) in PARTICLES and clean_alpha(tokens[-2]) in PARTICLES:
        compound3 = clean_alpha(tokens[-3]) + clean_alpha(tokens[-2]) + last_clean
        if compound3 in last_set:
            last_clean = compound3
            tokens = tokens[:-2]
    elif len(tokens) >= 3 and clean_alpha(tokens[-2]) in PARTICLES:
        compound2 = clean_alpha(tokens[-2]) + last_clean
        if compound2 in last_set:
            last_clean = compound2
            tokens = tokens[:-1]

    if last_clean not in last_set and '-' in raw_last:
        parts = [clean_alpha(p) for p in raw_last.split('-') if clean_alpha(p)]
        if parts and parts[-1] in last_set:
            last_clean = parts[-1]
        elif parts and parts[0] in last_set:
            last_clean = parts[0]

    first_candidates = tokens[:-1]
    non_initials = [clean_alpha(t) for t in first_candidates if len(clean_alpha(t)) > 1]
    if non_initials:
        first_name = non_initials[0]
    elif nick_tokens:
        first_name = nick_tokens[0]
    elif first_candidates:
        first_name = clean_alpha(first_candidates[0])
    else:
        first_name = pd.NA

    f_out = first_name if (pd.notna(first_name) and len(first_name) > 1) else pd.NA
    l_out = last_clean if (pd.notna(last_clean) and last_clean != '') else pd.NA
    return pd.Series([f_out, l_out])

candidates_clean[['first_name', 'last_name']] = candidates_clean['candidate'].apply(parse_first_last)

# Create a full name "token" so that names can be correctly matched up later
  # I had to add this feature myself as the parsing generated in this cell was not sufficient
def full_name(row):
    if pd.isna(row['first_name']) and pd.isna(row['last_name']):
      return str(row['candidate'])
    if pd.isna(row['first_name']):
      return "_ " + row['last_name']
    if pd.isna(row['last_name']):
      return row['first_name'] + " _"
    return row['first_name'] + ' ' + row['last_name']

candidates_clean['full_name'] = candidates_clean.apply(full_name, axis=1)


# Merge demographic proportions and flag missing Census matches
candidates_enriched = (
    candidates_clean
    .merge(first_lookup, left_on='first_name', right_on='FIRST NAME', how='left')
    .merge(last_lookup, left_on='last_name', right_on='LAST NAME', how='left')
    .drop(columns=['FIRST NAME', 'LAST NAME'])
)

candidates_enriched['first_missing_census'] = candidates_enriched['first_pct_white'].isna().astype(int)
candidates_enriched['last_missing_census'] = candidates_enriched['last_pct_white'].isna().astype(int)

# Check match rates among major-party candidates
major = candidates_enriched[candidates_enriched['party_simplified'].isin(['DEMOCRAT', 'REPUBLICAN'])]
print(f"Major-party first name match rate: {1 - major['first_missing_census'].mean():.2%}")
print(f"Major-party last name match rate:  {1 - major['last_missing_census'].mean():.2%}")




"""Step 3: Feature Engineer Democrat Vote Percentage Label"""

# Before we create the matchups and the vote % label, we need to ID all of the seats by year
  # so that we can identify the winner of the races that the candidates were put in
# This will become very important later
def seat_id(row):
    if row['office'] == 'US HOUSE':
        return "HOUSE", row['district']
    if not row['special']:
        return "SENATE", row['year'] % 6
    return "SENATE", row['year']

candidates_enriched['seat_id'] = candidates_enriched.apply(seat_id, axis=1)

# Now gather the winners of each race
race_identifiers = ['year', 'state', 'state_po', 'office', 'district', 'seat_id', 'special']

all_winners = (
    candidates_enriched
    .sort_values('candidatevotes', ascending = False)
    .groupby(race_identifiers, as_index = False)
    .first()[race_identifiers + ['full_name']]
    .rename(columns={'full_name':'winner_name'})
)

# Get the previous winners for the non-special elections
regular_winners = (
    all_winners[~all_winners['special']]
    .sort_values(['state', 'seat_id', 'year'])
    .copy()
)
regular_winners['prev_winner'] = regular_winners.groupby(['state', 'seat_id'])['winner_name'].shift(1)

# Filter out third-party candidates
major_candidates = candidates_enriched[
    candidates_enriched['party_simplified'].isin(['DEMOCRAT', 'REPUBLICAN'])
].copy()

# Group candidates of the same exact race together, then keep only the candidates
# running 1-on-1 against a member of the opposite party
counts = major_candidates.groupby(race_identifiers + ['party_simplified']).size().unstack(fill_value=0)
valid_races = counts[(counts['DEMOCRAT'] == 1) & (counts['REPUBLICAN'] == 1)].index

# Pull out all information on the valid candidates and de-index them
valid_candidates = major_candidates.set_index(race_identifiers).loc[valid_races].reset_index()

# Create two tables for democrats/republicans in the valid races,
dem = valid_candidates[valid_candidates['party_simplified'] == 'DEMOCRAT'].copy()
rep = valid_candidates[valid_candidates['party_simplified'] == 'REPUBLICAN'].copy()
# Then glue the races together, with candidate features separated by Democrat vs. Republican
matchups = pd.merge(
    dem, rep,
    on=race_identifiers,
    suffixes=('_dem', '_rep')
)
# Finally, calculate the Democrat vote share we've been trying to feature engineer the whole time
matchups['dem_two_party_share'] = (
    matchups['candidatevotes_dem'] / (matchups['candidatevotes_dem'] + matchups['candidatevotes_rep'])
)

"""Step 4: Feature Engineer Incumbency"""

# Glue on the previous seat winner for every race
matchups = matchups.merge(
    regular_winners[['year', 'state', 'seat_id', 'prev_winner']],
    on=['year', 'state', 'seat_id'],
    how='left'
)

# Also record Senate special election winners so they are labeled incumbent the next time a normal Senate election is held
special_sen = all_winners[(all_winners['office'] == 'US SENATE') & (all_winners['special'])]
special_lookup = {(row['state'], row['year']): row['winner_name'] for _, row in special_sen.iterrows()}

def compute_incumbency(row):
    prior_winners = {row['prev_winner']}
    if row['office'] == 'US SENATE':
        for possible_special in (2, 4):
            if (row['state'], row['year'] - possible_special) in special_lookup:
                prior_winners.add(special_lookup[(row['state'], row['year'] - possible_special)])
    prior_winners.discard(pd.NA)

    if row['full_name_dem'] in prior_winners:
        return 1
    if row['full_name_rep'] in prior_winners:
        return -1
    return 0

matchups['incumbency'] = matchups.apply(compute_incumbency, axis=1)




"""Step 5: Feature Engineer Partisanship Controls"""

# Create feature for previous Democratic vote share and fill with the previous election results
matchups = matchups.sort_values(['state', 'seat_id', 'year']).reset_index(drop=True)
matchups['prev_dem_share'] = matchups.groupby(['state', 'seat_id'])['dem_two_party_share'].shift(1)
# If there aren't previous results, put in 50%
  # Previously this was done with state averages, but that's a data leak! Oops!
matchups['prev_share_missing'] = matchups['prev_dem_share'].isna().astype(int)
matchups['prev_dem_share'] = matchups['prev_dem_share'].fillna(0.5)

""" REMOVED AND PUT LATER BECAUSE OF DATA LEAKAGE
# Replace rare/unknown names with the average value of all names
for col in ['first_pct_white_dem', 'first_pct_white_rep',
            'last_pct_white_dem', 'last_pct_white_rep',
            'first_pct_male_dem', 'first_pct_male_rep']:
    matchups[col] = matchups[col].fillna(matchups[col].mean())

# Calculate the differentials between the Democrat candidate's name and the Republican candidate's name
matchups['diff_first_white'] = matchups['first_pct_white_dem'] - matchups['first_pct_white_rep']
matchups['diff_last_white'] = matchups['last_pct_white_dem'] - matchups['last_pct_white_rep']
matchups['diff_first_male'] = matchups['first_pct_male_dem'] - matchups['first_pct_male_rep']
"""

# Add in the control flags for Senate elections and special elections
matchups['is_senate'] = (matchups['office'] == 'US SENATE').astype(int)
matchups['is_special'] = matchups['special'].astype(int)

print("Total contested two-party matchups:", len(matchups))




"""Step 6: Split Train & Test Sets"""

# Create a graph of candidates as nodes and edges connecting their opps
G = nx.Graph()
for _, row in matchups.iterrows():
    # note: state and full name is used because john smith ohio =/= john smith tennessee
    # why not state + district? well, candidates will often upgrade or get redistricted
    dem_cand = (row['state'], row['full_name_dem'])
    rep_cand = (row['state'], row['full_name_rep'])
    G.add_edge(dem_cand, rep_cand)

# Give each connected component an ID and plug that ID into the matchups
cand_to_component = {}
for component_idx, component in enumerate(nx.connected_components(G)):
    for cand in component:
        cand_to_component[cand] = component_idx

matchups['component_id'] = matchups.apply(
    lambda r: cand_to_component[(r['state'], r['full_name_dem'])],
    axis=1
)

# Testing the number of components
print("Number of unique candidates:", G.number_of_nodes())
print("Number of connected components:", matchups['component_id'].nunique())

# Perform the split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
train_idx, test_idx = next(gss.split(matchups, matchups['dem_two_party_share'], groups=matchups['component_id']))

train_data = matchups.iloc[train_idx].copy()
test_data = matchups.iloc[test_idx].copy()

# Impute missing Census frequencies here so that there isn't contamination from the test data
  # (Previously it was in Step 6 before the split, but that's data leakage!)
raw_name_cols = [
    'first_pct_white_dem', 'first_pct_white_rep',
    'last_pct_white_dem', 'last_pct_white_rep',
    'first_pct_male_dem', 'first_pct_male_rep'
]
train_name_means = train_data[raw_name_cols].mean()

for df in (train_data, test_data):
    df[raw_name_cols] = df[raw_name_cols].fillna(train_name_means)
    df['diff_first_white'] = df['first_pct_white_dem'] - df['first_pct_white_rep']
    df['diff_last_white'] = df['last_pct_white_dem'] - df['last_pct_white_rep']
    df['diff_first_male'] = df['first_pct_male_dem'] - df['first_pct_male_rep']

# Check to make sure nothing leaked
train_candidates = set(zip(train_data['state'], train_data['full_name_dem'])) | set(zip(train_data['state'], train_data['full_name_rep']))
test_candidates = set(zip(test_data['state'], test_data['full_name_dem'])) | set(zip(test_data['state'], test_data['full_name_rep']))

print("Train rows:", len(train_data), "Percent:", len(train_data)/len(matchups))
print("Test rows:", len(test_data), "Percent:", len(test_data)/len(matchups))
print("Overlapping candidates between train and test:", len(train_candidates & test_candidates))




"""Step 7: Run the Model!"""

# List the numerical and categorical features
numerical = [
    'diff_first_white', 'diff_last_white', 'diff_first_male',
    'first_missing_census_dem', 'first_missing_census_rep',
    'last_missing_census_dem', 'last_missing_census_rep',
    'incumbency', 'prev_dem_share', 'is_senate', 'is_special',
    'prev_share_missing'
]
categorical = ['state', 'year']

# Ensure categorical types are regular strings
for df in (train_data, test_data):
    df['state'] = df['state'].astype(str)
    df['year'] = df['year'].astype(str)

# Establish train and test features + labels
x_train = train_data[numerical + categorical]
y_train = train_data['dem_two_party_share'].astype(float)
x_test = test_data[numerical + categorical]
y_test = test_data['dem_two_party_share'].astype(float)

# Preprocessing to standardize scales for numericals and one-hot encode state/year
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
])

# Build the pipeline from preprocessing to Ridge regression
pipe = Pipeline([
    ('prep', preprocessor),
    ('ridge', Ridge())
])

# Create a 5-fold on the data so we don't need to have a specific validation set
kfold = GroupKFold(n_splits=5)

# Create a grid of a bunch of different alphas so we can test out which one is the best
grid = GridSearchCV(
    pipe,
    param_grid={'ridge__alpha': [0.1, 1.0, 10.0, 50.0, 100.0, 500.0]},
    cv=kfold,
    scoring='neg_mean_squared_error'
)
grid.fit(x_train, y_train, groups=train_data['component_id'])

y_pred = grid.predict(x_test)
print("Best Alpha:", grid.best_params_['ridge__alpha'])
print("Test R-Squared:", r2_score(y_test, y_pred))
print("Test RMSE:", root_mean_squared_error(y_test, y_pred))

"""**FINAL WEIGHTS**"""

# Get best pipeline & weights
best_pipe = grid.best_estimator_
coefs = best_pipe.named_steps['ridge'].coef_
stds = best_pipe.named_steps['prep'].named_transformers_['num'].scale_

# Create summary statistics for the results
coef_table = pd.DataFrame({
    'Feature': numerical,
    'Standardized Weight (per 1 SD)': coefs[:len(numerical)],
    'Raw Effect (per 1.0 unit change)': coefs[:len(numerical)] / stds
})

print(coef_table.to_string(index=False))




"""DIAGNOSTICS AND SUBGROUP ANALYSIS"""

# Remove higher-level alpha values from the GridSearch options
grid_low_alpha = GridSearchCV(
    pipe,
    param_grid={'ridge__alpha': [0.01, 0.1, 1.0]},
    cv=kfold,
    scoring='neg_mean_squared_error'
)
grid_low_alpha.fit(x_train, y_train, groups=train_data['component_id'])

low_alpha_pred = grid_low_alpha.predict(x_test)
print("Best Low Alpha:", grid_low_alpha.best_params_['ridge__alpha'])
print("Low Alpha R-Squared:", r2_score(y_test, low_alpha_pred))
print("Low Alpha RMSE:", root_mean_squared_error(y_test, low_alpha_pred))

# Get best pipeline & weights
pipe_low_alpha = grid_low_alpha.best_estimator_
coefs_low_alpha = pipe_low_alpha.named_steps['ridge'].coef_
stds_low_alpha = pipe_low_alpha.named_steps['prep'].named_transformers_['num'].scale_

# Create summary statistics for the results
coef_low_alpha = pd.DataFrame({
    'Feature': numerical,
    'Standardized Weight (per 1 SD)': coefs_low_alpha[:len(numerical)],
    'Raw Effect (per 1.0 unit change)': coefs_low_alpha[:len(numerical)] / stds_low_alpha
})

print(coef_low_alpha.to_string(index=False))


print("\n\n\n=== SUBGROUP ANALYSIS ===")

# Copy test data so we can calculate error and make some subgroups
test_eval = test_data.copy()
test_eval['y_pred'] = y_pred
test_eval['abs_err'] = (test_eval['dem_two_party_share'] - test_eval['y_pred']).abs()

subgroups = {
    'All Tested Races': test_eval,
    'No Incumbent Races': test_eval[test_eval['incumbency'] == 0],
    'Incumbent Races': test_eval[test_eval['incumbency'] != 0],
    'House Only': test_eval[test_eval['is_senate'] == 0],
    'Senate Only': test_eval[test_eval['is_senate'] == 1],
    'No 1976 (no 50% imputation)': test_eval[test_eval['year'].astype(int) > 1976],
}

# Get the statistical correlation data on each subgroup to see if any is more performant
subgroup_rows = []
for name, subgroup in subgroups.items():
    subgroup_rows.append({
        'Subgroup': name,
        'N': len(subgroup),
        'R^2': r2_score(subgroup['dem_two_party_share'], subgroup['y_pred']),
        'RMSE': root_mean_squared_error(subgroup['dem_two_party_share'], subgroup['y_pred']),
        'MAE': subgroup['abs_err'].mean()
    })
print(pd.DataFrame(subgroup_rows).to_string(index=False))




"""OTHER MODEL INVESTIGATIONS"""

# Get the tested features and controls separated for the ablation test
name_features = [
    'diff_first_white', 'diff_last_white', 'diff_first_male',
    'first_missing_census_dem', 'first_missing_census_rep',
    'last_missing_census_dem', 'last_missing_census_rep'
]
control_numerical = [c for c in numerical if c not in name_features]

# Create state_year interaction column to test what I originally put in my proposal
for df in (train_data, test_data):
    df['state_year'] = df['state'] + '_' + df['year']

results=[]

# Mean dummy regression
dummy = DummyRegressor(strategy='mean').fit(x_train, y_train)
dummy_pred = dummy.predict(x_test)
results.append(('Mean Dummy', np.nan, r2_score(y_test, dummy_pred), root_mean_squared_error(y_test, dummy_pred)))

# Prev-based prediction
pb_pred = x_test['prev_dem_share'].values
results.append(('Prev-Based (prev_dem_share)', np.nan, r2_score(y_test, pb_pred), root_mean_squared_error(y_test, pb_pred)))

# Ablate tested features
ctrl_prep = ColumnTransformer([
    ('num', StandardScaler(), control_numerical),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
])
ctrl_grid = GridSearchCV(
    Pipeline([('prep', ctrl_prep), ('ridge', Ridge())]),
    param_grid={'ridge__alpha': [0.1, 1.0, 10.0, 50.0, 100.0, 500.0]},
    cv=kfold, scoring='neg_mean_squared_error'
).fit(train_data[control_numerical + categorical], y_train, groups=train_data['component_id'])
pred_ctrl = ctrl_grid.predict(test_data[control_numerical + categorical])
results.append(('Ridge (No Names)', ctrl_grid.best_params_['ridge__alpha'], r2_score(y_test, pred_ctrl), root_mean_squared_error(y_test, pred_ctrl)))

# My final design
results.append(('Ridge (With Names)', grid.best_params_['ridge__alpha'], r2_score(y_test, y_pred), root_mean_squared_error(y_test, y_pred)))

# My proposed inclusion of state-year one-hot vector
sy_prep = ColumnTransformer([
    ('num', StandardScaler(), numerical),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['state', 'year', 'state_year'])
])
sy_grid = GridSearchCV(
    Pipeline([('prep', sy_prep), ('ridge', Ridge())]),
    param_grid={'ridge__alpha': [1.0, 10.0, 50.0, 100.0, 500.0]},
    cv=kfold, scoring='neg_mean_squared_error'
).fit(train_data[numerical + ['state', 'year', 'state_year']], y_train, groups=train_data['component_id'])
sy_pred = sy_grid.predict(test_data[numerical + ['state', 'year', 'state_year']])
results.append(('Ridge (With state-year)', sy_grid.best_params_['ridge__alpha'], r2_score(y_test, sy_pred), root_mean_squared_error(y_test, sy_pred)))


comparison_table = pd.DataFrame(results, columns=['Model', 'Best Alpha', 'Test R^2', 'Test RMSE'])
print(comparison_table.to_string(index=False))