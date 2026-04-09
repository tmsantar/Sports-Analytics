library(cfbfastR)
library(dplyr)

Sys.setenv(CFBD_API_KEY = "ebRGLD1n7bbRBvUvOtzhkd274HsG366L9Q1S/pr6F28q4I6bTTABDrEYQWKGMqQH")

rosters <- cfbd_team_roster(2025)


rosters_clean <- rosters %>%
  mutate(
    player_name = paste(firstName, lastName)
  ) %>%
  select(
    athlete_id,
    player_name,
    team,
    position,
    headshot_url
  ) %>%
  distinct(athlete_id, .keep_all = TRUE)

write.csv(rosters_clean, "rosters_clean.csv", row.names = FALSE)
