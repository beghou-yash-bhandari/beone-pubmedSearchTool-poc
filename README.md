# Introduction 
This is a template starter project for building Streamlit apps. The template is ready to go with Auth0-based Single Sign-On (SSO) and deployment to Azure App Services. 

You can copy this repo to your own project, provision an Azure App Service and Auth0 application, add your own code to app.py, and be ready to show your demo app in a very short amount of time. I created this template so that we can keep our code in-house in Azure DevOps repos and deployed with proper authentication. My hope is that this is a quick-to-deploy alternative to using private GitHub repos and hosting on streamlit.app.

# What is Streamlit?
Streamlit (https://streamlit.io/) is a free and open-source Python framework for quickly building simple data-oriented apps with little to no front-end web development experience required. The Streamlit docs are well written and have walkthroughs for a number of common needs. Streamlit has an assortment of common, reactive UI widgets, easy-to-use session state and caching, and a clean UI style. It is easy to start from scratch or convert existing python scripts to a Streamlit app for data visualization, AI chat, or simple calculators. You can focus on building your idea rather than HTML, JavaScript, and CSS.

# Getting Started
This code is tested on Python 3.11. Ensure you are running Python 3.11 or later.

## Copy this repo to a new repo in your DevOps project
1. Create a new Git repo in your Azure DevOps project and clone the empty repo locally. (target)
2. Clone this repo to a separate local location. (source)
3. Copy the code from the source local repo to the target local repo. 
4. Make an initial commit in your new repo.

## Provision an Azure App Service & Auth0 application
Contact Chris Weathers and Jason Keglovitz to provision these resources. Be sure to note that you're creating a Streamlit app from this template and include the following in your request:
- Project name
- URL of your Git repo in Azure DevOps. (This is the clone URL that you used in step 2 above.)

When the resources are ready, they will send you the following info:
- URL where your app will be deployed to
- AUTH0_CLIENTID
- AUTH0_DOMAIN
- AUTH0_AUDIENCE

## Prepare your local dev environment
1. Open your local folder in VS Code or the IDE of your choice.
2. Create and prepare your virtual environment (commands below are from the project root)
    1. Windows Command Prompt

            pip install virtualenv
            virtualenv venv
            venv\Scripts\activate
            python.exe -m pip install --upgrade pip
            pip install -r requirements.txt

    2. Mac Terminal

            python venv -m venv (or python3 venv -m venv if that's your python alias)
            source venv/bin/activate
            python -m pip install --upgrade pip
            pip install -r requirements.txt

3. Create an empty .env file in the root of your project to store your environment variables, such as the Auth0 settings that you'll get in the next section
4. Using the AUTH0 settings from the previous section, paste the following into your empty .env file and update each value.

        AUTH0_CLIENTID = "**********"
        AUTH0_DOMAIN = "**********"
        AUTH0_AUDIENCE = "**********"

## Test SSO locally
1. From your command prompt/terminal, run the streamlit app

        streamlit run app.py

2. Your browser should automatically open up to http://localhost:8501. Click the login button and test that SSO works as expected.

## Deploy to Azure App Service
Committing your code to the main branch will automatically deploy code to your Azure App Service. It typically takes about 5 minutes to complete the deployment after the commit. 

If SSO is working locally, you can commit what you have and test it out on the URL you were given above. From here on, you can add code to your app, commit, and deploy as needed.

# Contribute
Do not commit code to this repo. If you have suggestions for improving this template, contat Jason Keglovitz via email.

# Appendix
## Azure App Service Configuration
- Deployment Center -> Connect to Azure Repos and then choose the correct organization, project, and repository.
- Configuration -> General Settings -> Create a Python 3.11 stack unless something else is needed
- Configuration -> General Settings -> Startup command should be:

        python3 -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0

- Configuration -> Application Settings -> Add application settings for the 3 AUTH0 settings.
    - AUTH0_AUDIENCE = https://beghou-prod.us.auth0.com/api/v2/
    - AUTH0_DOMAIN = sso.beghouconsulting.com
    - AUTH0_CLIENTID = the Client Id obtained when creating the Auth0 application

## Auth0 Configuration
- Create applications in the production tentant.
- Application Type = Single Page Application
- Allowed Callback URLs & Allowed Logout URLs have the same value (be sure to replace app service domain)

        http://localhost:8501/component/auth0_component.login_button/index.html, https://{app service domain}.azurewebsites.net/component/auth0_component.login_button/index.html

- Allowed Web Origins value (be sure to replace app service domain)

        http://localhost:8501, https://{app service domain}.azurewebsites.net

- Enable the BeghouADFS connection for the application