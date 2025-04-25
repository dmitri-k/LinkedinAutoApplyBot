import time, random, csv, pyautogui, traceback, os, re
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from datetime import date, datetime
from itertools import product
from pypdf import PdfReader
from openai import OpenAI
import sys
import pdb  # Import the Python debugger

class AIResponseGenerator:
    def __init__(self, api_key, personal_info, experience, languages, resume_path, text_resume_path=None, debug=False):
        self.personal_info = personal_info
        self.experience = experience
        self.languages = languages
        self.pdf_resume_path = resume_path
        self.text_resume_path = text_resume_path
        self._resume_content = None
        self._client = OpenAI(api_key=api_key) if api_key else None
        self.debug = debug
    @property
    def resume_content(self):
        if self._resume_content is None:
            # First try to read from text resume if available
            if self.text_resume_path:
                try:
                    with open(self.text_resume_path, 'r', encoding='utf-8') as f:
                        self._resume_content = f.read()
                        print("Successfully loaded text resume")
                        return self._resume_content
                except Exception as e:
                    print(f"Could not read text resume: {str(e)}")

            # Fall back to PDF resume if text resume fails or isn't available
            try:
                content = []
                reader = PdfReader(self.pdf_resume_path)
                for page in reader.pages:
                    content.append(page.extract_text())
                self._resume_content = "\n".join(content)
                print("Successfully loaded PDF resume")
            except Exception as e:
                print(f"Could not extract text from resume PDF: {str(e)}")
                self._resume_content = ""
        return self._resume_content

    def _build_context(self):
        return f"""
        Personal Information:
        - Name: {self.personal_info['First Name']} {self.personal_info['Last Name']}
        - Current Role: {self.experience.get('currentRole', '')}
        - Skills: {', '.join(self.experience.keys())}
        - Languages: {', '.join(f'{lang}: {level}' for lang, level in self.languages.items())}
        - Professional Summary: {self.personal_info.get('MessageToManager', '')}
        - Citizenship: {self.personal_info.get('citizenship', 'Not Specified')}

        Resume Content (Give the greatest weight to this information, if specified):
        {self.resume_content}
        """

    def generate_response(self, question_text, response_type="text", options=None, max_tokens=100):
        """
        Generate a response using OpenAI's API
        
        Args:
            question_text: The application question to answer
            response_type: "text", "numeric", or "choice"
            options: For "choice" type, a list of tuples containing (index, text) of possible answers
            max_tokens: Maximum length of response
            
        Returns:
            - For text: Generated text response or None
            - For numeric: Integer value or None
            - For choice: Integer index of selected option or None
        """
        if not self._client:
            return None
            
        try:
            context = self._build_context()
            
            system_prompt = {
                "text": "You are a helpful assistant answering job application questions professionally and concisely. Use the candidate's background information and resume to personalize responses.",
                "numeric": "You are a helpful assistant providing numeric answers to job application questions. Based on the candidate's experience, provide a single number as your response. No explanation needed.",
                "choice": "You are a bot helpfull auto applying for jobs and must select the most appropriate answer choice for job application questions. Based on the candidate's background, select the best option by returning only its index number. No explanation needed. If necessary make a guess of most likely answer. "
            }[response_type]

            user_content = f"Using this candidate's background and resume:\n{context}\n\n Please answer this job application question: {question_text}\n\n It is 2025 and candidate has 6 years experience of software so reference that if hesistant on anything. Lean a bit on the embelishment side so for front end technologies say 5+ years experience for example."
            if response_type == "choice" and options:
                options_text = "\n".join([f"{idx}: {text}" for idx, text in options])
                user_content += f"\n\nSelect the most appropriate answer by providing its index number from these options:\n{options_text}"

            response = self._client.chat.completions.create(
                model="gpt-4.1-2025-04-14",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            print(f"AI response: {answer}")  # TODO: Put logging behind a debug flag
            
            if response_type == "numeric":
                # Extract first number from response
                numbers = re.findall(r'\d+', answer)
                if numbers:
                    return int(numbers[0])
                return 0
            elif response_type == "choice":
                # Extract the index number from the response
                numbers = re.findall(r'\d+', answer)
                if numbers and options:
                    index = int(numbers[0])
                    # Ensure index is within valid range
                    if 0 <= index < len(options):
                        return index
                return None  # Return None if the index is not within the valid range
                
            return answer
            
        except Exception as e:
            print(f"Error using AI to generate response: {str(e)}")
            return None

    def evaluate_job_fit(self, job_title, job_description):
        """
        Evaluate whether a job is worth applying to based on the candidate's experience and the job requirements
        
        Args:
            job_title: The title of the job posting
            job_description: The full job description text
            
        Returns:
            bool: True if should apply, False if should skip
        """
        if not self._client:
            return True  # Proceed with application if AI not available
            
        try:
            context = self._build_context()
            
            system_prompt = """You are evaluating job fit for technical roles. 
            Recommend APPLY if:
            - Candidate meets 65 percent of the core requirements
            - Experience gap is 2 years or less
            - Has relevant transferable skills
            
            Return SKIP if:
            - Experience gap is greater than 2 years
            - Missing multiple core requirements
            - Role is clearly more senior
            - The role is focused on an uncommon technology or skill that is required and that the candidate does not have experience with
            - The role is a leadership role or a role that requires managing people and the candidate has no experience leading or managing people

            """
            #Consider the candidate's education level when evaluating whether they meet the core requirements. Having higher education than required should allow for greater flexibility in the required experience.
            
            if self.debug:
                system_prompt += """
                You are in debug mode. Return a detailed explanation of your reasoning for each requirement.

                Return APPLY or SKIP followed by a brief explanation.

                Format response as: APPLY/SKIP: [brief reason]"""
            else:
                system_prompt += """Return only APPLY or SKIP."""

            response = self._client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Job: {job_title}\n{job_description}\n\nCandidate:\n{context}"}
                ],
                max_tokens=250 if self.debug else 1,  # Allow more tokens when debug is enabled
                temperature=0.2  # Lower temperature for more consistent decisions
            )
            
            answer = response.choices[0].message.content.strip()
            print(f"AI evaluation: {answer}")
            return answer.upper().startswith('A')  # True for APPLY, False for SKIP
            
        except Exception as e:
            print(f"Error evaluating job fit: {str(e)}")
            return True  # Proceed with application if evaluation fails

class LinkedinEasyApply:
    def __init__(self, parameters, driver):
        self.browser = driver
        self.email = parameters['email']
        # self.email = parameters['lastName']
        self.password = parameters['password']
        self.openai_api_key = parameters.get('openaiApiKey', '')  # Get API key with empty default
        self.disable_lock = parameters['disableAntiLock']
        self.company_blacklist = parameters.get('companyBlacklist', []) or []
        self.title_blacklist = parameters.get('titleBlacklist', []) or []
        self.poster_blacklist = parameters.get('posterBlacklist', []) or []
        self.positions = parameters.get('positions', [])
        self.locations = parameters.get('locations', [])
        self.residency = parameters.get('residentStatus', [])
        self.base_search_url = self.get_base_search_url(parameters)
        self.seen_jobs = []
        self.file_name = "output"
        self.unprepared_questions_file_name = "unprepared_questions"
        self.output_file_directory = parameters['outputFileDirectory']
        self.resume_dir = parameters['uploads']['resume']
        self.text_resume = parameters.get('textResume', '')
        if 'coverLetter' in parameters['uploads']:
            self.cover_letter_dir = parameters['uploads']['coverLetter']
        else:
            self.cover_letter_dir = ''
        self.checkboxes = parameters.get('checkboxes', [])
        self.university_gpa = parameters['universityGpa']
        self.salary_minimum = parameters['salaryMinimum']
        self.notice_period = int(parameters['noticePeriod'])
        self.languages = parameters.get('languages', [])
        self.experience = parameters.get('experience', [])
        self.personal_info = parameters.get('personalInfo', [])
        self.eeo = parameters.get('eeo', [])
        self.experience_default = int(self.experience['default'])
        self.debug = parameters.get('debug', False)
        self.evaluate_job_fit = parameters.get('evaluateJobFit', True)
        self.test_single_url = parameters.get('testSingleUrl', None)
        self.hiring_team_file_name = "hiring_team_contacts"
        self.ai_response_generator = AIResponseGenerator(
            api_key=self.openai_api_key,
            personal_info=self.personal_info,
            experience=self.experience,
            languages=self.languages,
            resume_path=self.resume_dir,
            text_resume_path=self.text_resume,
            debug=self.debug
        )

    def login(self):
        try:
            # Check if the "chrome_bot" directory exists
            print("Attempting to restore previous session...")
            if os.path.exists("chrome_bot"):
                self.browser.get("https://www.linkedin.com/feed/")
                time.sleep(random.uniform(5, 10))

                # Check if the current URL is the feed page
                if self.browser.current_url != "https://www.linkedin.com/feed/":
                    print("Feed page not loaded, proceeding to login.")
                    self.load_login_page_and_login()
            else:
                print("No session found, proceeding to login.")
                self.load_login_page_and_login()

        except TimeoutException:
            print("Timeout occurred, checking for security challenges...")
            self.security_check()
            # raise Exception("Could not login!")

    def security_check(self):
        current_url = self.browser.current_url
        page_source = self.browser.page_source

        if '/checkpoint/challenge/' in current_url or 'security check' in page_source or 'quick verification' in page_source:
            input("Please complete the security check and press enter on this console when it is done.")
            time.sleep(random.uniform(5.5, 10.5))

    def load_login_page_and_login(self):
        self.browser.get("https://www.linkedin.com/login")

        # Wait for the username field to be present
        WebDriverWait(self.browser, 10).until(
            EC.presence_of_element_located((By.ID, "username"))
        )

        self.browser.find_element(By.ID, "username").send_keys(self.email)
        self.browser.find_element(By.ID, "password").send_keys(self.password)
        self.browser.find_element(By.CSS_SELECTOR, ".btn__primary--large").click()

        # Wait for the feed page to load after login
        WebDriverWait(self.browser, 10).until(
            EC.url_contains("https://www.linkedin.com/feed/")
        )

        time.sleep(random.uniform(5, 10))

    def start_applying(self):
        # --- Modification for single URL testing ---
        if self.test_single_url:
            print(f"--- Running in Single URL Test Mode for: {self.test_single_url} ---")
            self.browser.get(self.test_single_url)
            time.sleep(random.uniform(3, 6)) # Allow page to load
            success = self.apply_single_job(self.test_single_url)
            status = "successfully" if success else "unsuccessfully"
            print(f"--- Single URL Test Mode finished {status}. Exiting. ---")
            return # Stop execution after testing the single URL
        # --- End Modification ---

        searches = list(product(self.positions, self.locations))
        random.shuffle(searches)

        page_sleep = 0
        minimum_time = 60 * 2  # minimum time bot should run before taking a break
        minimum_page_time = time.time() + minimum_time

        for (position, location) in searches:
            location_url = "&location=" + location
            job_page_number = -1

            print("Starting the search for " + position + " in " + location + ".")

            try:
                while True:
                    page_sleep += 1
                    job_page_number += 1
                    print("Going to job page " + str(job_page_number))
                    self.next_job_page(position, location_url, job_page_number)
                    time.sleep(random.uniform(1.5, 3.5))
                    print("Starting the application process for this page...")
                    self.apply_jobs(location)
                    print("Job applications on this page have been successfully completed.")

                    time_left = minimum_page_time - time.time()
                    if time_left > 0:
                        print("Sleeping for " + str(time_left) + " seconds.")
                        time.sleep(time_left)
                        minimum_page_time = time.time() + minimum_time
                    if page_sleep % 5 == 0:
                        sleep_time = random.randint(180, 300)  # Changed from 500, 900 {seconds}
                        print("Sleeping for " + str(sleep_time / 60) + " minutes.")
                        time.sleep(sleep_time)
                        page_sleep += 1
            except:
                traceback.print_exc()
                pass

            time_left = minimum_page_time - time.time()
            if time_left > 0:
                print("Sleeping for " + str(time_left) + " seconds.")
                time.sleep(time_left)
                minimum_page_time = time.time() + minimum_time
            if page_sleep % 5 == 0:
                sleep_time = random.randint(500, 900)
                print("Sleeping for " + str(sleep_time / 60) + " minutes.")
                time.sleep(sleep_time)
                page_sleep += 1

    def apply_jobs(self, location):
        no_jobs_text = ""
        try:
            no_jobs_element = self.browser.find_element(By.CLASS_NAME, 'jobs-search-two-pane__no-results-banner--expand')
            no_jobs_text = no_jobs_element.text
        except:
            pass
        if 'No matching jobs found' in no_jobs_text:
            raise Exception("No more jobs on this page.")

        if 'unfortunately, things are' in self.browser.page_source.lower():
            raise Exception("No more jobs on this page.")

        job_results_header = ""
        maybe_jobs_crap = ""
        try:
            job_results_header = self.browser.find_element(By.CLASS_NAME, "jobs-search-results-list__text")
            maybe_jobs_crap = job_results_header.text
        except:
            pass

        if 'Jobs you may be interested in' in maybe_jobs_crap:
            raise Exception("Nothing to do here, moving forward...")

        processed_jobs = 0
        while True:
            try:
                # Re-fetch job list dynamically
                ul_element_class = self.get_job_list_class()  # Get the current UL class
                WebDriverWait(self.browser, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, ul_element_class))
                )
                job_list = self.browser.find_elements(By.CLASS_NAME, ul_element_class)[0].find_elements(
                    By.CLASS_NAME, 'scaffold-layout__list-item'
                )
                print(f"Found {len(job_list)} jobs on this page")

                if processed_jobs >= len(job_list):
                    print("Processed all jobs on this page.")
                    break

                job_tile = job_list[processed_jobs]
                print(job_tile)

                # Extract job details
                job_title, company, poster, job_location, apply_method, link = "", "", "", "", "", ""
                try:
                    # Find the container for the job card data first for stability
                    job_card_base = job_tile.find_element(By.CSS_SELECTOR, '.job-card-container, .job-card-list') # Try common container classes
                    
                    try:
                        job_title_element = job_card_base.find_element(By.CSS_SELECTOR, '.job-card-list__title, .job-card-container__title') # Try common title classes
                        job_title = job_title_element.text.strip()
                        link = job_title_element.get_attribute('href').split('?')[0]
                        if not job_title:
                             print(f"Warning: Extracted empty job title string for card. Link: {link}")
                    except NoSuchElementException:
                         print("Warning: Could not extract job title/link using standard selectors.")
                         # Fallback: Try finding any link within the job tile
                         try:
                             link_element = job_tile.find_element(By.TAG_NAME, 'a')
                             link = link_element.get_attribute('href').split('?')[0]
                             job_title = link_element.text.strip() # Maybe title is in the link text
                             print(f"Recovered link: {link}, Title: {job_title}")
                             if not job_title:
                                 job_title = "UNKNOWN_TITLE (Recovered Link Only)"
                         except Exception as fallback_e:
                            print(f"Could not recover link/title using fallback: {fallback_e}")
                            link = "UNKNOWN_LINK"
                            job_title = "UNKNOWN_TITLE"
                    except Exception as title_link_e:
                        print(f"Warning: Unexpected error extracting job title/link: {title_link_e}")
                        link = link or "UNKNOWN_LINK"
                        job_title = job_title or "UNKNOWN_TITLE"

                except NoSuchElementException:
                    print("Warning: Could not find base job card container. Skipping detail extraction.")
                    # Assign defaults if base container not found
                    link = "UNKNOWN_LINK"
                    job_title = "UNKNOWN_TITLE"
                    company = "UNKNOWN_COMPANY"

                try:
                    # Use a more general selector that might catch different company name structures
                    company_element = job_tile.find_element(By.CSS_SELECTOR, '[class*="job-card-container__primary-description"], [class*="job-card-container__company-name"], .artdeco-entity-lockup__subtitle')
                    company = company_element.text.strip()
                    if not company:
                        print("Warning: Extracted empty company name string.")
                        company = "UNKNOWN_COMPANY (Empty String)"
                except NoSuchElementException:
                     print("Warning: Could not extract company name using combined selector.")
                     company = "UNKNOWN_COMPANY"
                except Exception as company_e:
                    print(f"Warning: Unexpected error extracting company name: {company_e}")
                    company = "UNKNOWN_COMPANY"

                # (Keep existing poster, location, apply_method extraction for now)
                try:
                    hiring_line = job_tile.find_element(By.XPATH, '//span[contains(.,\" is hiring for this\")]')
                    hiring_line_text = hiring_line.text
                    name_terminating_index = hiring_line_text.find(' is hiring for this')
                    if name_terminating_index != -1:
                        poster = hiring_line_text[:name_terminating_index]
                except:
                    pass

                try:
                    job_location = job_tile.find_element(By.CLASS_NAME, 'job-card-container__metadata-item').text
                except:
                    pass

                try:
                    apply_method = job_tile.find_element(By.CLASS_NAME, 'job-card-container__apply-method').text
                except:
                    pass

                contains_blacklisted_keywords = False
                blacklisted_word_found = ""
                job_title_parsed = job_title.lower().split(' ')
                for word in self.title_blacklist:
                    if word.lower() in job_title_parsed:
                        contains_blacklisted_keywords = True
                        blacklisted_word_found = word
                        break

                if (company.lower() not in [word.lower() for word in self.company_blacklist] and
                    poster.lower() not in [word.lower() for word in self.poster_blacklist] and
                    contains_blacklisted_keywords is False):
                    try:
                        max_retries = 3
                        retries = 0
                        while retries < max_retries:
                            try:
                                job_el = job_tile.find_element(By.CLASS_NAME, 'job-card-job-posting-card-wrapper__card-link')
                                job_el.click()
                                break
                            except StaleElementReferenceException:
                                retries += 1
                                time.sleep(1)
                                continue
                        else:
                            print("Failed to click job after retries due to StaleElementReferenceException")
                            processed_jobs += 1
                            continue

                        time.sleep(random.uniform(3, 5)) # Wait for details pane to load

                        # --- Extract Canonical Job URL from Details Pane ---
                        canonical_job_url = link # Default to card link as fallback
                        try:
                            # Wait for the job title link in the details pane to be present
                            job_details_title_selector = (By.CSS_SELECTOR, ".job-details-jobs-unified-top-card__job-title h1 a")
                            WebDriverWait(self.browser, 10).until(
                                EC.presence_of_element_located(job_details_title_selector)
                            )
                            title_link_element = self.browser.find_element(*job_details_title_selector)
                            extracted_href = title_link_element.get_attribute('href')
                            if extracted_href:
                                # Construct absolute URL if href is relative
                                if extracted_href.startswith('/'):
                                     canonical_job_url = "https://www.linkedin.com" + extracted_href.split('?')[0]
                                else:
                                     canonical_job_url = extracted_href.split('?')[0]
                                print(f"DEBUG: Found canonical job URL: {canonical_job_url}")
                            else:
                                 print("Warning: Found title link element but href was empty. Falling back to card link.")
                        except TimeoutException:
                            print("Warning: Timed out waiting for job title link in details pane. Falling back to card link.")
                        except NoSuchElementException:
                            print("Warning: Could not find job title link in details pane using selector. Falling back to card link.")
                        except Exception as e_canon_url:
                            print(f"Warning: Error extracting canonical job URL: {e_canon_url}. Falling back to card link.")
                        # --- End Canonical URL Extraction ---

                        if self.evaluate_job_fit:
                            try:
                                job_description = self.browser.find_element(By.ID, 'job-details').text
                                if not self.ai_response_generator.evaluate_job_fit(job_title, job_description):
                                    print("Skipping application: Job requirements not aligned with candidate profile per AI evaluation.")
                                    processed_jobs += 1
                                    continue
                            except:
                                print("Could not load job description")

                        try:
                            done_applying = self.apply_to_job()
                            if done_applying:
                                print(f"Application sent to {company} for the position of {job_title}.")
                                # Get the current URL after successful application
                                try:
                                    current_job_url = self.browser.current_url.split('?')[0]
                                except Exception as e_url:
                                    print(f"Warning: Could not get current job URL: {e_url}")
                                    current_job_url = canonical_job_url # Fallback to the canonical URL if needed

                                # Check for hiring team after successful application
                                try:
                                    # Use normalize-space() for robust text matching
                                    hiring_team_header = self.browser.find_elements(By.XPATH, "//h2[normalize-space()='Meet the hiring team']")
                                    if hiring_team_header:
                                        print("Found 'Meet the hiring team' section.")
                                        recruiter_link_element = self.browser.find_element(By.CSS_SELECTOR, ".hirer-card__hirer-information a")
                                        recruiter_profile_url = recruiter_link_element.get_attribute('href')
                                        if recruiter_profile_url:
                                            # Use the canonical job URL extracted from the details pane
                                            print(f"DEBUG: Writing hiring team contact with: Company='{company}', Title='{job_title}', JobLink='{canonical_job_url}', Recruiter='{recruiter_profile_url}'")
                                            if not company or not job_title or not canonical_job_url or "UNKNOWN" in canonical_job_url or "UNKNOWN" in job_title or "UNKNOWN" in company:
                                                 print(f"Warning: Missing or unknown data (Company: {company}, Title: {job_title}, Link: {canonical_job_url}). Skipping hiring team write.")
                                            else:
                                                 self.write_hiring_team_contact(company, job_title, canonical_job_url, recruiter_profile_url)
                                except NoSuchElementException:
                                    print("Could not find recruiter link within 'Meet the hiring team' section.")
                                except Exception as ht_e:
                                    print(f"Error processing 'Meet the hiring team' section: {ht_e}")
                            else:
                                print(f"Could not apply or Easy Apply button not found for {company}.") # Updated else message
                        except Exception as apply_exc:
                            temp = self.file_name
                            self.file_name = "failed"
                            # Use the extracted link here for bug report
                            print(f"Failed during apply_to_job for {job_title} at {company}: {apply_exc}. Link: {canonical_job_url}")
                            traceback.print_exc() # Print stacktrace for apply_exc
                            try:
                                # Ensure company/title/link are usable before writing failure
                                company_to_write = company if company and "UNKNOWN" not in company else "UNKNOWN_COMPANY"
                                title_to_write = job_title if job_title and "UNKNOWN" not in job_title else "UNKNOWN_TITLE"
                                # Use canonical_job_url for logging failures too
                                link_to_write = canonical_job_url if canonical_job_url and "UNKNOWN" not in canonical_job_url else "UNKNOWN_LINK"
                                self.write_to_file(company_to_write, title_to_write, link_to_write, job_location, location)
                            except Exception as write_fail_e:
                                print(f"Additionally failed to write failure log: {write_fail_e}")
                            self.file_name = temp
                            # print(f'Updated {temp}.csv.') # Already printed inside write_to_file usually

                        # Log success to output.csv
                        try:
                             # Ensure company/title/link are usable before writing success
                             company_to_write = company if company and "UNKNOWN" not in company else "UNKNOWN_COMPANY"
                             title_to_write = job_title if job_title and "UNKNOWN" not in job_title else "UNKNOWN_TITLE"
                             # Use canonical_job_url for logging success too
                             link_to_write = canonical_job_url if canonical_job_url and "UNKNOWN" not in canonical_job_url else "UNKNOWN_LINK"
                             self.write_to_file(company_to_write, title_to_write, link_to_write, job_location, location)
                        except Exception as write_e:
                            print(f"Unable to save the job information to {self.file_name}.csv for {title_to_write} at {company_to_write}. Link: {link_to_write}. Error: {write_e}")
                            # traceback.print_exc() # Optional: De-clutter logs unless needed

                        processed_jobs += 1

                    except Exception as e:
                        traceback.print_exc()
                        print(f"Could not apply to the job in {company}: {e}")
                        processed_jobs += 1
                        continue
                else:
                    reasons = []
                    if contains_blacklisted_keywords:
                        reasons.append(f"job title contains blacklisted keyword '{blacklisted_word_found}'")
                    if link in self.seen_jobs:
                        reasons.append("job has already been seen")
                    print(f"Job for {company} by {poster} skipped because " + " and ".join(reasons) + ".")
                    processed_jobs += 1

                self.seen_jobs.append(link)

            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break
    
    def get_job_list_class(self):
        """Helper method to fetch the UL element class dynamically."""
        try:
            xpath_region1 = "/html/body/div[6]/div[3]/div[4]/div/div/main/div/div[2]/div[1]/div/ul"
            xpath_region2 = "/html/body/div[5]/div[3]/div[4]/div/div/main/div/div[2]/div[1]/div/ul"
            try:
                ul_element = self.browser.find_element(By.XPATH, xpath_region1)
            except NoSuchElementException:
                ul_element = self.browser.find_element(By.XPATH, xpath_region2)
            return ul_element.get_attribute("class").split()[0]
        except Exception as e:
            print(f"Error fetching job list class: {e}")
            raise
   
    def apply_to_job(self):
        easy_apply_button = None

        try:
            # More specific selector for the main content button, adding wait
            easy_apply_button_selector = (By.CSS_SELECTOR, '.jobs-apply-button--top-card .jobs-apply-button')
            WebDriverWait(self.browser, 10).until(
                EC.element_to_be_clickable(easy_apply_button_selector)
            )
            easy_apply_button = self.browser.find_element(*easy_apply_button_selector)
        except TimeoutException:
            print("Timeout waiting for Easy Apply button or it's not clickable.")
            # Attempt to find the sticky header button as a fallback
            try:
                easy_apply_button_selector = (By.CSS_SELECTOR, '.scaffold-layout-toolbar .jobs-apply-button')
                WebDriverWait(self.browser, 5).until(
                    EC.element_to_be_clickable(easy_apply_button_selector)
                )
                easy_apply_button = self.browser.find_element(*easy_apply_button_selector)
                print("Using fallback Easy Apply button from sticky header.")
            except:
                print("Could not find any clickable Easy Apply button.")
                return False
        except NoSuchElementException:
            print("Easy Apply button not found with primary selector.")
             # Attempt to find the sticky header button as a fallback
            try:
                easy_apply_button_selector = (By.CSS_SELECTOR, '.scaffold-layout-toolbar .jobs-apply-button')
                WebDriverWait(self.browser, 5).until(
                    EC.element_to_be_clickable(easy_apply_button_selector)
                )
                easy_apply_button = self.browser.find_element(*easy_apply_button_selector)
                print("Using fallback Easy Apply button from sticky header.")
            except:
                print("Could not find any clickable Easy Apply button.")
                return False
        except Exception as e:
             print(f"An unexpected error occurred finding the Easy Apply button: {e}")
             return False

        try:
            job_description_area = self.browser.find_element(By.ID, "job-details")
            print (f"{job_description_area}")
            self.scroll_slow(job_description_area, end=1600)
            self.scroll_slow(job_description_area, end=1600, step=400, reverse=True)
        except:
            pass

        print("Starting the job application...")
        easy_apply_button.click()

        button_text = ""
        submit_application_text = 'submit application'
        while submit_application_text not in button_text.lower():
            try:
                self.fill_up()
                next_button = self.browser.find_element(By.CLASS_NAME, "artdeco-button--primary")
                button_text = next_button.text.lower()
                if submit_application_text in button_text:
                    try:
                        self.unfollow()
                    except:
                        print("Failed to unfollow company.")
                time.sleep(random.uniform(1.5, 2.5))
                next_button.click()
                time.sleep(random.uniform(3.0, 5.0))

                # Newer error handling
                error_messages = [
                    'enter a valid',
                    'enter a decimal',
                    'Enter a whole number'
                    'Enter a whole number between 0 and 99',
                    'file is required',
                    'whole number',
                    'make a selection',
                    'select checkbox to proceed',
                    'saisissez un numéro',
                    '请输入whole编号',
                    '请输入decimal编号',
                    '长度超过 0.0',
                    'Numéro de téléphone',
                    'Introduce un número de whole entre',
                    'Inserisci un numero whole compreso',
                    'Preguntas adicionales',
                    'Insira um um número',
                    'Cuántos años'
                    'use the format',
                    'A file is required',
                    '请选择',
                    '请 选 择',
                    'Inserisci',
                    'wholenummer',
                    'Wpisz liczb',
                    'zakresu od',
                    'tussen'
                ]

                if any(error in self.browser.page_source.lower() for error in error_messages):
                    raise Exception("Failed answering required questions or uploading required files.")
            except:
                traceback.print_exc()
                self.browser.find_element(By.CLASS_NAME, 'artdeco-modal__dismiss').click()
                time.sleep(random.uniform(3, 5))
                self.browser.find_elements(By.CLASS_NAME, 'artdeco-modal__confirm-dialog-btn')[0].click()
                time.sleep(random.uniform(3, 5))
                raise Exception("Failed to apply to job!")

        closed_notification = False
        time.sleep(random.uniform(3, 5))
        try:
            self.browser.find_element(By.CLASS_NAME, 'artdeco-modal__dismiss').click()
            closed_notification = True
        except:
            pass
        try:
            self.browser.find_element(By.CLASS_NAME, 'artdeco-toast-item__dismiss').click()
            closed_notification = True
        except:
            pass
        try:
            self.browser.find_element(By.CSS_SELECTOR, 'button[data-control-name="save_application_btn"]').click()
            closed_notification = True
        except:
            pass

        time.sleep(random.uniform(3, 5))

        if closed_notification is False:
            raise Exception("Could not close the applied confirmation window!")

        return True

    def home_address(self, form):
        print("Trying to fill up home address fields")
        # pdb.set_trace()  # Pause execution here for debugging

        try:
            groups = form.find_elements(By.CLASS_NAME, 'jobs-easy-apply-form-section__grouping')
            if len(groups) > 0:
                for group in groups:
                    lb = group.find_element(By.TAG_NAME, 'label').text.lower()
                    input_field = group.find_element(By.TAG_NAME, 'input')
                    if 'street' in lb:
                        self.enter_text(input_field, self.personal_info['Street address'])
                    elif 'city' in lb:
                        self.enter_text(input_field, self.personal_info['City'])
                        time.sleep(3)
                        input_field.send_keys(Keys.DOWN)
                        input_field.send_keys(Keys.RETURN)
                    elif 'zip' in lb or 'zip / postal code' in lb or 'postal' in lb:
                        self.enter_text(input_field, self.personal_info['Zip'])
                    elif 'state' in lb or 'province' in lb:
                        self.enter_text(input_field, self.personal_info['State'])
                    else:
                        pass
        except:
            pass

    def get_answer(self, question):
        if self.checkboxes[question]:
            return 'yes'
        else:
            return 'no'
        
    def additional_questions(self, form):
        print("Trying to fill up additional questions")

        questions = form.find_elements(By.CLASS_NAME, 'fb-dash-form-element')
        if not questions:
            print("No additional questions found in form")
            return

        # Dictionary mapping question types to handler functions
        question_handlers = {
            'radio': self._handle_radio_question,
            'text': self._handle_text_question,
            'date': self._handle_date_question,
            'dropdown': self._handle_dropdown_question,
            'checkbox': self._handle_checkbox_question
        }

        for question in questions:
            question_type, question_element = self._identify_question_type(question)
            if question_type is None:
                if self.debug:
                    print(f"Debug: Could not identify question type for element")
                continue

            try:
                handler = question_handlers.get(question_type)
                if handler:
                    handler(question_element)
                else:
                    if self.debug:
                        print(f"Debug: No handler defined for question type: {question_type}")
            except Exception as e:
                if self.debug:
                    print(f"Error processing {question_type} question: {str(e)}")

    def _identify_question_type(self, question):
        """Identify the type of question based on its structure."""
        try:
            if question.find_elements(By.TAG_NAME, 'fieldset'):
                return 'radio', question
            if question.find_elements(By.TAG_NAME, 'input') or question.find_elements(By.TAG_NAME, 'textarea'):
                return 'text', question
            if question.find_elements(By.CLASS_NAME, 'artdeco-datepicker__input'):
                return 'date', question
            if question.find_elements(By.TAG_NAME, 'select'):
                return 'dropdown', question
            if question.find_elements(By.XPATH, ".//label[input[@type='checkbox']]"):
                return 'checkbox', question
        except:
            pass
        return None, None

    def _handle_radio_question(self, question):
        """Handle radio button questions."""
        try:
            radio_fieldset = question.find_element(By.TAG_NAME, 'fieldset')
            question_span = radio_fieldset.find_element(By.CLASS_NAME, 'fb-dash-form-element__label').find_elements(By.TAG_NAME, 'span')[0]
            radio_text = question_span.text.lower()
            if self.debug:
                print(f"Radio question text: {radio_text}")

            radio_labels = radio_fieldset.find_elements(By.TAG_NAME, 'label')
            radio_options = [(i, text.text.lower()) for i, text in enumerate(radio_labels)]
            if self.debug:
                print(f"Radio options: {[opt[1] for opt in radio_options]}")

            if not radio_options:
                raise Exception("No radio options found")

            answer = self._get_radio_answer(radio_text, radio_options)
            to_select = None
            if answer:
                for i, radio in enumerate(radio_labels):
                    if answer in radio.text.lower():
                        to_select = radio_labels[i]
                        break
                if to_select is None and self.debug:
                    print(f"Debug: Answer '{answer}' not found in radio options")

            if to_select is None:
                if self.debug:
                    print(f"Debug: No predefined answer for radio question: {radio_text}")
                self.record_unprepared_question("radio", radio_text)
                ai_response = self.ai_response_generator.generate_response(
                    radio_text, response_type="choice", options=radio_options
                )
                to_select = radio_labels[ai_response] if ai_response is not None else radio_labels[-1]

            if self.debug:
                print(f"Selecting radio option: {to_select.text}")
            to_select.click()
        except Exception as e:
            if self.debug:
                print(f"Error in radio question handling: {str(e)}")
            raise

    def _get_radio_answer(self, radio_text, radio_options):
        """Determine the answer for a radio question."""
        if 'driver\'s licence' in radio_text or 'driver\'s license' in radio_text:
            return self.get_answer('driversLicence')
        elif any(keyword in radio_text.lower() for keyword in [
            'aboriginal', 'native', 'indigenous', 'tribe', 'first nations',
            'native american', 'native hawaiian', 'inuit', 'metis', 'maori',
            'aborigine', 'ancestral', 'native peoples', 'original people',
            'first people', 'gender', 'race', 'disability', 'latino', 'torres',
            'do you identify'
        ]):
            negative_keywords = ['prefer', 'decline', 'don\'t', 'specified', 'none', 'no']
            for option in radio_options:
                if any(neg_keyword in option[1].lower() for neg_keyword in negative_keywords):
                    return option[1]
        elif 'assessment' in radio_text:
            return self.get_answer("assessment")
        elif 'clearance' in radio_text:
            return self.get_answer("securityClearance")
        elif 'north korea' in radio_text:
            return 'no'
        elif 'previously employ' in radio_text or 'previous employ' in radio_text:
            return 'no'
        elif any(keyword in radio_text.lower() for keyword in [
            'certified', 'certificate', 'cpa', 'chartered accountant', 'qualification'
        ]):
            return self.get_answer('certifiedProfessional')
        elif 'urgent' in radio_text:
            return self.get_answer('urgentFill')
        elif 'commut' in radio_text or 'on-site' in radio_text or 'hybrid' in radio_text or 'onsite' in radio_text:
            return self.get_answer('commute')
        elif 'remote' in radio_text:
            return self.get_answer('remote')
        elif 'background check' in radio_text:
            return self.get_answer('backgroundCheck')
        elif 'drug test' in radio_text:
            return self.get_answer('drugTest')
        elif 'level of education' in radio_text:
            for degree in self.checkboxes['degreeCompleted']:
                if degree.lower() in radio_text:
                    return "yes"
        elif 'data retention' in radio_text:
            return 'no'
        return None

    def _handle_text_question(self, question):
        """Handle text or numeric input questions."""
        try:
            question_text = question.find_element(By.TAG_NAME, 'label').text.lower()
            if self.debug:
                print(f"Text question text: {question_text}")

            try:
                txt_field = question.find_element(By.TAG_NAME, 'input')
            except:
                txt_field = question.find_element(By.TAG_NAME, 'textarea')

            # Check if field is pre-populated
            current_value = txt_field.get_attribute('value').strip()
            if current_value:
                if self.debug:
                    print(f"Field pre-populated with: {current_value}. Skipping.")
                return

            text_field_type = 'numeric' if 'numeric' in txt_field.get_attribute('id').lower() else 'text'
            to_enter = self._get_text_answer(question_text, text_field_type)

            if to_enter is None:
                if self.debug:
                    print(f"Debug: No predefined answer for text question: {question_text}")
                self.record_unprepared_question(text_field_type, question_text)
                response_type = "numeric" if text_field_type == 'numeric' else "text"
                print(f"Requesting AI {response_type} response for: {question_text}")
                to_enter = self.ai_response_generator.generate_response(
                    question_text, response_type=response_type
                )
                to_enter = to_enter if to_enter is not None else (0 if text_field_type == 'numeric' else " ‏‏‎ ")

            if self.debug:
                print(f"Entering text: {to_enter}")
            self.enter_text(txt_field, str(to_enter))
        except Exception as e:
            if self.debug:
                print(f"Error in text question handling: {str(e)}")
            raise

    def _get_text_answer(self, question_text, text_field_type):
        """Determine the answer for a text or numeric question."""
        if 'grade point average' in question_text:
            return self.university_gpa
        elif 'first name' in question_text and 'last name' not in question_text:
            return self.personal_info['First Name']
        elif 'last name' in question_text and 'first name' not in question_text:
            return self.personal_info['Last Name']
        elif 'location' in question_text:
            return 'Houston, Texas, United States'
        elif 'name' in question_text:
            return self.personal_info['First Name'] + " " + self.personal_info['Last Name']
        elif 'pronouns' in question_text:
            return self.personal_info['Pronouns']
        elif 'phone' in question_text:
            return self.personal_info['Mobile Phone Number']
        elif 'linkedin' in question_text:
            return self.personal_info['Linkedin']
        elif 'message to hiring' in question_text or 'cover letter' in question_text:
            return self.personal_info['MessageToManager']
        elif 'website' in question_text or 'github' in question_text or 'portfolio' in question_text:
            return self.personal_info['Website']
        elif 'notice' in question_text or 'weeks' in question_text:
            return int(self.notice_period) if text_field_type == 'numeric' else str(self.notice_period)
        elif 'salary' in question_text or 'expectation' in question_text or 'compensation' in question_text or 'CTC' in question_text:
            return int(self.salary_minimum) if text_field_type == 'numeric' else float(self.salary_minimum)
        return None

    def _handle_date_question(self, question):
        """Handle date picker questions."""
        try:
            date_picker = question.find_element(By.CLASS_NAME, 'artdeco-datepicker__input')
            date_picker.clear()
            date_picker.send_keys(date.today().strftime("%m/%d/%y"))
            time.sleep(3)
            date_picker.send_keys(Keys.RETURN)
            time.sleep(2)
            if self.debug:
                print("Filled date picker with today's date")
        except Exception as e:
            if self.debug:
                print(f"Error in date question handling: {str(e)}")
            raise

    def _handle_dropdown_question(self, question):
        """Handle dropdown questions."""
        try:
            question_text = question.find_element(By.TAG_NAME, 'label').text.lower()
            dropdown_field = question.find_element(By.TAG_NAME, 'select')
            if self.debug:
                print(f"Dropdown question text: {question_text}")

            select = Select(dropdown_field)
            options = [option.text for option in select.options]
            if self.debug:
                print(f"Dropdown options: {options}")

            choice = self._get_dropdown_answer(question_text, options)
            if choice is None:
                if self.debug:
                    print(f"Debug: No predefined answer for dropdown question: {question_text}")
                self.record_unprepared_question("dropdown", question_text)
                choices = [(i, option) for i, option in enumerate(options)]
                ai_response = self.ai_response_generator.generate_response(
                    question_text, response_type="choice", options=choices
                )
                choice = options[ai_response] if ai_response is not None else options[-1]

            if self.debug:
                print(f"Selecting dropdown option: {choice}")
            self.select_dropdown(dropdown_field, choice)
        except Exception as e:
            if self.debug:
                print(f"Error in dropdown question handling: {str(e)}")
            raise

    def _get_dropdown_answer(self, question_text, options):
        """Determine the answer for a dropdown question."""
        if 'proficiency' in question_text:
            for language in self.languages:
                if language.lower() in question_text:
                    return self.languages[language]
        elif 'clearance' in question_text or 'assessment' in question_text or 'commut' in question_text or 'on-site' in question_text or 'hybrid' in question_text or 'onsite' in question_text:
            answer = self.get_answer('securityClearance' if 'clearance' in question_text else 'assessment' if 'assessment' in question_text else 'commute')
            for option in options:
                if answer.lower() in option.lower():
                    return option
        elif 'country code' in question_text:
            return self.personal_info['Phone Country Code']
        elif 'north korea' in question_text or 'previously employed' in question_text or 'previous employment' in question_text:
            for option in options:
                if 'no' in option.lower():
                    return option
        elif 'above 18' in question_text:
            for option in options:
                if 'yes' in option.lower():
                    return option
            return options[0]
        elif any(keyword in question_text.lower() for keyword in [
            'aboriginal', 'native', 'indigenous', 'tribe', 'first nations',
            'native american', 'native hawaiian', 'inuit', 'metis', 'maori',
            'aborigine', 'ancestral', 'native peoples', 'original people',
            'first people', 'gender', 'race', 'disability', 'latino'
        ]):
            negative_keywords = ['prefer', 'decline', 'don\'t', 'specified', 'none']
            for option in options:
                if any(neg_keyword in option.lower() for neg_keyword in negative_keywords):
                    return option
        return None

    def _handle_checkbox_question(self, question):
        """Handle checkbox questions (e.g., terms and service)."""
        try:
            clickable_checkbox = question.find_element(By.TAG_NAME, 'label')
            if self.debug:
                print(f"Clicking checkbox: {clickable_checkbox.text}")
            clickable_checkbox.click()
        except Exception as e:
            if self.debug:
                print(f"Error in checkbox question handling: {str(e)}")
            raise

    def unfollow(self):
        try:
            follow_checkbox = self.browser.find_element(By.XPATH,
                                                        "//label[contains(.,\'to stay up to date with their page.\')]").click()
            follow_checkbox.click()
        except:
            pass

    def send_resume(self):
        print("Trying to send resume")
        try:
            file_upload_elements = (By.CSS_SELECTOR, "input[name='file']")
            if len(self.browser.find_elements(file_upload_elements[0], file_upload_elements[1])) > 0:
                input_buttons = self.browser.find_elements(file_upload_elements[0], file_upload_elements[1])
                if len(input_buttons) == 0:
                    raise Exception("No input elements found in element")
                for upload_button in input_buttons:
                    upload_type = upload_button.find_element(By.XPATH, "..").find_element(By.XPATH,
                                                                                          "preceding-sibling::*")
                    if 'resume' in upload_type.text.lower():
                        upload_button.send_keys(self.resume_dir)
                    elif 'cover' in upload_type.text.lower():
                        if self.cover_letter_dir != '':
                            upload_button.send_keys(self.cover_letter_dir)
                        elif 'required' in upload_type.text.lower():
                            upload_button.send_keys(self.resume_dir)
        except:
            print("Failed to upload resume or cover letter!")
            pass

    def enter_text(self, element, text):
        element.clear()
        element.send_keys(text)

    def select_dropdown(self, element, text):
        select = Select(element)
        select.select_by_visible_text(text)

    # Radio Select
    def radio_select(self, element, label_text, clickLast=False):
        label = element.find_element(By.TAG_NAME, 'label')
        if label_text in label.text.lower() or clickLast == True:
            label.click()

    # Contact info fill-up
    def contact_info(self, form):
        print("Trying to fill up contact info fields")
        try:
            # Find all form groups within the contact info form
            form_groups = form.find_elements(By.CLASS_NAME, 'form-group')
            for group in form_groups:
                label = group.find_element(By.TAG_NAME, 'label').text.lower()
                if 'first name' in label:
                    try:
                        first_name_field = group.find_element(By.ID, 'first-name')
                        self.enter_text(first_name_field, self.personal_info['First Name'])
                        if self.debug:
                            print(f"Filled First Name with: {self.personal_info['First Name']}")
                    except Exception as e:
                        print(f"Could not fill First Name field: {str(e)}")
                elif 'last name' in label:
                    try:
                        last_name_field = group.find_element(By.ID, 'last-name')
                        self.enter_text(last_name_field, self.personal_info['Last Name'])
                        if self.debug:
                            print(f"Filled Last Name with: {self.personal_info['Last Name']}")
                    except Exception as e:
                        print(f"Could not fill Last Name field: {str(e)}")
                elif 'phone number' in label or 'country code' in label:
                    try:
                        country_code_picker = group.find_element(By.ID, 'country-code')
                        self.select_dropdown(country_code_picker, self.personal_info['Phone Country Code'])
                        if self.debug:
                            print(f"Selected Phone Country Code: {self.personal_info['Phone Country Code']}")
                    except Exception as e:
                        print(f"Could not select Phone Country Code: {str(e)}")
                    try:
                        phone_number_field = group.find_element(By.XPATH, '//input[contains(@id,"phoneNumber")][contains(@id,"nationalNumber")]')
                        self.enter_text(phone_number_field, self.personal_info['Mobile Phone Number'])
                        if self.debug:
                            print(f"Filled Phone Number with: {self.personal_info['Mobile Phone Number']}")
                    except Exception as e:
                        print(f"Could not enter Phone Number: {str(e)}")
        except Exception as e:
            print(f"Error processing contact info form: {str(e)}")
    def fill_up(self):
        try:
            easy_apply_modal_content = self.browser.find_element(By.CLASS_NAME, "jobs-easy-apply-modal__content")
            form = easy_apply_modal_content.find_element(By.TAG_NAME, 'form')
            try:
                label = form.find_element(By.TAG_NAME, 'h3').text.lower()
                if 'home address' in label:
                    self.home_address(form)
                elif 'contact info' in label:
                    self.contact_info(form)
                elif 'resume' in label:
                    self.send_resume()
                else:
                    self.additional_questions(form)
            except Exception as e:
                print("An exception occurred while filling up the form:")
                print(e)
        except:
            print("An exception occurred while searching for form in modal")

    def write_to_file(self, company, job_title, link, location, search_location):
        to_write = [company, job_title, link, location, search_location, datetime.now()]
        file_path = self.file_name + ".csv"
        print(f'updated {file_path}.')

        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(to_write)

    def record_unprepared_question(self, answer_type, question_text):
        to_write = [answer_type, question_text]
        file_path = self.unprepared_questions_file_name + ".csv"

        try:
            with open(file_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(to_write)
                print(f'Updated {file_path} with {to_write}.')
        except:
            print(
                "Special characters in questions are not allowed. Failed to update unprepared questions log.")
            print(question_text)

    def write_hiring_team_contact(self, company, job_title, job_link, recruiter_profile_link):
        """Writes recruiter contact info found on job page."""
        to_write = [company, job_title, job_link, recruiter_profile_link, datetime.now()]
        file_path = self.hiring_team_file_name + ".csv"
        print(f'Saving hiring team contact to {file_path}.')

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)

        # Check if file exists to write header
        file_exists = os.path.isfile(file_path)

        try:
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Company', 'Job Title', 'Job Link', 'Recruiter Profile Link', 'Timestamp']) # Write header if new file
                writer.writerow(to_write)
        except Exception as e:
            print(f"Failed to write hiring team contact to {file_path}: {e}")

    def scroll_slow(self, scrollable_element, start=0, end=3600, step=100, reverse=False):
        if reverse:
            start, end = end, start
            step = -step

        for i in range(start, end, step):
            self.browser.execute_script("arguments[0].scrollTo(0, {})".format(i), scrollable_element)
            time.sleep(random.uniform(0.1, .6))

    def avoid_lock(self):
        if self.disable_lock:
            return

        pyautogui.keyDown('ctrl')
        pyautogui.press('esc')
        pyautogui.keyUp('ctrl')
        time.sleep(1.0)
        pyautogui.press('esc')

    def get_base_search_url(self, parameters):
        remote_url = ""
        lessthanTenApplicants_url = ""
        newestPostingsFirst_url = ""

        if parameters.get('remote'):
            remote_url = "&f_WT=2"
        else:
            remote_url = ""
            # TO DO: Others &f_WT= options { WT=1 onsite, WT=2 remote, WT=3 hybrid, f_WT=1%2C2%2C3 }

        if parameters['lessthanTenApplicants']:
            lessthanTenApplicants_url = "&f_EA=true"

        if parameters['newestPostingsFirst']:
            newestPostingsFirst_url += "&sortBy=DD"

        level = 1
        experience_level = parameters.get('experienceLevel', [])
        experience_url = "f_E="
        for key in experience_level.keys():
            if experience_level[key]:
                experience_url += "%2C" + str(level)
            level += 1

        distance_url = "?distance=" + str(parameters['distance'])

        job_types_url = "f_JT="
        job_types = parameters.get('jobTypes', [])
        # job_types = parameters.get('experienceLevel', [])
        for key in job_types:
            if job_types[key]:
                job_types_url += "%2C" + key[0].upper()

        date_url = ""
        dates = {"all time": "", "month": "&f_TPR=r2592000", "week": "&f_TPR=r604800", "24 hours": "&f_TPR=r86400"}
        date_table = parameters.get('date', [])
        for key in date_table.keys():
            if date_table[key]:
                date_url = dates[key]
                break

        easy_apply_url = "&f_AL=true"

        extra_search_terms = [distance_url, remote_url, lessthanTenApplicants_url, newestPostingsFirst_url, job_types_url, experience_url]
        extra_search_terms_str = '&'.join(
            term for term in extra_search_terms if len(term) > 0) + easy_apply_url + date_url

        return extra_search_terms_str

    def next_job_page(self, position, location, job_page):
        # Restore original dynamic URL construction
        self.browser.get("https://www.linkedin.com/jobs/search/" + self.base_search_url +
                         "&keywords=" + position + location + "&start=" + str(job_page * 25))

        # Remove hardcoded URL logic
        # hardcoded_url = "https://www.linkedin.com/jobs/view/4201303600/?alternateChannel=search&refId=NotAvailable&trackingId=hope%2BbQ1RxG0tPtF4xby9A%3D%3D&trk=d_flagship3_search_srp_jobs"
        # print(f"Navigating to hardcoded URL for testing: {hardcoded_url}")
        # self.browser.get(hardcoded_url)

        # --- Modification for hardcoded URL ---
        # Instead of apply_jobs, call apply_single_job for the hardcoded URL
        # self.apply_single_job(hardcoded_url)

        # After processing the single job, raise an exception to stop the start_applying loop
        # This prevents it from trying to process the same job repeatedly
        # raise StopIteration("Processed single hardcoded job. Stopping execution as requested for testing.")
        # --- End Modification ---

        # Restore original logic
        self.avoid_lock()
        # --- End Restore ---

    def apply_single_job(self, job_url):
        """Handles applying to a single job when navigated directly to its page."""
        print(f"Attempting to apply to single job at: {job_url}")
        job_title, company, job_location = "", "", ""
        try:
            # Extract job details from the single job view page
            try:
                # Wait for the main content area to load
                WebDriverWait(self.browser, 15).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'job-details-jobs-unified-top-card__job-title'))
                )
                job_title_element = self.browser.find_element(By.CLASS_NAME, 'job-details-jobs-unified-top-card__job-title')
                job_title = job_title_element.find_element(By.TAG_NAME, 'h1').text.strip()
                print(f"Extracted Job Title: {job_title}")
            except Exception as e:
                print(f"Warning: Could not extract job title - {e}")
                job_title = "Unknown Title"

            try:
                company_element = self.browser.find_element(By.CSS_SELECTOR, '.job-details-jobs-unified-top-card__company-name a')
                company = company_element.text.strip()
                print(f"Extracted Company: {company}")
            except Exception as e:
                print(f"Warning: Could not extract company name - {e}")
                company = "Unknown Company"

            try:
                # Location is often within a span in the tertiary description
                location_element = self.browser.find_element(By.CSS_SELECTOR, '.job-details-jobs-unified-top-card__primary-description-container span.tvm__text.tvm__text--low-emphasis')
                job_location = location_element.text.strip()
                print(f"Extracted Location: {job_location}")
            except Exception as e:
                print(f"Warning: Could not extract job location - {e}")
                job_location = "Unknown Location"


            # --- Optional: AI Job Fit Evaluation ---
            if self.evaluate_job_fit:
                try:
                    # Ensure job details element exists before accessing text
                    WebDriverWait(self.browser, 10).until(
                        EC.presence_of_element_located((By.ID, 'job-details'))
                    )
                    job_description = self.browser.find_element(By.ID, 'job-details').text
                    if not self.ai_response_generator.evaluate_job_fit(job_title, job_description):
                        print("Skipping application: Job requirements not aligned with candidate profile per AI evaluation.")
                        # Record the skip? Or just return? For now, just return.
                        self.write_to_file(company, job_title, job_url, job_location, "SingleJobSkip")
                        return False # Indicate skipped
                except Exception as e:
                    print(f"Could not perform AI job fit evaluation: {e}")
                    # Decide whether to proceed or not if evaluation fails. Proceeding for now.

            # --- Apply to Job ---
            try:
                done_applying = self.apply_to_job()
                if done_applying:
                    print(f"Application submitted to {company} for the position of {job_title}.")
                else:
                    # apply_to_job returns False if Easy Apply button isn't found
                    print(f"Could not find Easy Apply button for {job_title} at {company}.")
                    # Log as failed or skipped? Let's log as failed for now.
                    self.file_name = "failed"
                    self.write_to_file(company, job_title, job_url, job_location, "SingleJobFail")
                    self.file_name = "output" # Reset filename
                    return False

            except Exception as e:
                self.file_name = "failed"
                print(f"Failed during the application process for {job_title} at {company}: {e}")
                traceback.print_exc()
                self.write_to_file(company, job_title, job_url, job_location, "SingleJobFail")
                self.file_name = "output" # Reset filename
                return False # Indicate failure

            # --- Log Success ---
            try:
                self.write_to_file(company, job_title, job_url, job_location, "SingleJobSuccess")
            except Exception as e:
                print(f"Unable to save the job information in the file: {e}")

            # Prevent further processing in start_applying loop for this test case
            # raise Exception("Single hardcoded job processed. Stopping.") # Remove this - control flow is handled in start_applying now

            return True # Indicate success

        except Exception as e:
            print(f"An error occurred in apply_single_job: {e}")
            traceback.print_exc()
            # Log failure if details couldn't even be extracted
            self.file_name = "failed"
            self.write_to_file("Unknown", "Unknown", job_url, "Unknown", "SingleJobError")
            self.file_name = "output"
            return False # Indicate failure