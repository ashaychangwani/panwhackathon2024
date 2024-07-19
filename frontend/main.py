import time

import requests
import streamlit as st


def main():
    st.title("File Upload and Processing App")

    uploaded_file = st.file_uploader("Choose a file")
    objective = st.text_input("Enter your objective")

    if st.button("Submit"):
        if uploaded_file is not None and objective:
            token = uploaded_file.name  # Use the filename as a token
            data = {"video_url": token, "objective": objective}
            response = requests.post("http://localhost:8080/process", json=data)

            if response.status_code == 200:
                task_id = response.json()["task_id"]
                st.success("Data successfully sent to the server!")
                status_placeholder = st.empty()
                while True:
                    status_response = requests.get(
                        f"http://localhost:8080/status/{task_id}"
                    )
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        status_list = status_data.get("status", [])
                        status_text = "\n".join(
                            f"- {'✅' if item['status'] == 'completed' else '🔄'} {item['description']}"
                            for item in status_list
                        )
                        status_placeholder.markdown(status_text)
                        if len(status_list) > 0 and all(
                            item["status"] == "completed" for item in status_list
                        ):
                            st.success("Processing completed!")
                            print(status_list)
                            break
                    else:
                        st.error("Failed to get status from the server.")
                        break
                    time.sleep(1)
            else:
                st.error("Failed to send data to the server.")
        else:
            st.warning("Please upload a file and enter an objective.")


if __name__ == "__main__":
    main()
