class TaskStatus:
    tasks = []

    def append(self, message):
        self.tasks.append({"description": message, "status": "In Progress"})

    def complete(self, message):
        for task in self.tasks:
            if task["description"] == message:
                task["status"] = "completed"
                return
        self.tasks.append({"description": message, "status": "completed"})

    def delete_prefix(self, message):
        self.tasks = [
            task for task in self.tasks if not task["description"].startswith(message)
        ]

    def _summarize_task(self, prefix, summary):
        completed_count = sum(
            1
            for task in self.tasks
            if task["status"] == "completed" and task["description"].startswith(prefix)
        )
        in_progress_count = sum(
            1
            for task in self.tasks
            if task["status"] == "In Progress"
            and task["description"].startswith(prefix)
        )
        return {
            "description": f"{summary} {completed_count} segments so far",
            "status": "completed" if in_progress_count == 0 else "In Progress",
        }

    def get_tasks(self):
        output_tasks = []
        transcribed_added = False
        contextualized_added = False

        for task in self.tasks:
            if task["description"].startswith("Transcribing segment "):
                if not transcribed_added:
                    output_tasks.append(
                        self._summarize_task("Transcribing segment ", "Transcribed")
                    )
                    transcribed_added = True
            elif task["description"].startswith("Contextualizing frame "):
                if not contextualized_added:
                    output_tasks.append(
                        self._summarize_task("Contextualizing frame ", "Contextualized")
                    )
                    contextualized_added = True
            else:
                output_tasks.append(task)
        return output_tasks

    def finished(self, bool=True):
        self.tasks = [{"description": "completed", "status": "completed"}]

    def empty(self):
        self.tasks = []


tasks_status = dict()
