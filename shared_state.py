class TaskStatus:
    tasks = []

    def append(self, message):
        self.tasks.append({"description": message, "status": "In Progress"})

    def complete(self, message):
        for task in self.tasks:
            if task["description"] == message:
                task["status"] = "completed"
                break

    def delete_prefix(self, message):
        for task in self.tasks:
            if task["description"].startswith(message):
                self.tasks.remove(task)


tasks_status = dict()
