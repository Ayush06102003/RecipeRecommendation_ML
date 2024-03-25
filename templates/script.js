function addTask(event) {
    event.preventDefault(); // Prevent the default form submission behavior
    
    var taskInput = document.getElementById("taskInput");
    var taskList = document.getElementById("taskList");
    
    if (taskInput.value === "") {
        alert("Please enter an ingredient.");
        return;
    }
    
    var li = document.createElement("li");
    li.textContent = taskInput.value;
    
    li.onclick = function() {
        this.classList.toggle("completed");
    };
    
    taskList.appendChild(li);
    taskInput.value = "";
}
