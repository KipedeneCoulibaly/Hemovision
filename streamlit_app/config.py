"""

Config file for Streamlit App

"""

from streamlit_app.member import Member

TITLE = "Hemovision"

TEAM_MEMBERS = [
    Member(
        name="Bertrand-Elie DURAN",
        github_url="https://github.com/eliebd/",
    ),
    Member(
        name="Sergey SASNOUSKI",
        github_url="https://github.com/ssasnouski/",
    ),
    Member(
        name="Joseph LIEBER",
        linkedin_url="https://www.linkedin.com/in/joseph-lieber-687878a6/",
        github_url="https://github.com/josephlieber75/",
    ),
    Member(
        name="Kipédène Coulibaly",
        linkedin_url="https://www.linkedin.com/in/kip%C3%A9d%C3%A8necoulibaly/",
        github_url="https://github.com/KipedeneCoulibaly/",
    ),
]

PROMOTION = "Promotion Bootcamp Data Scientist - Juin 2023"
