class ModuleHijacker:
    def __init__(self, module):
        self.__module = module
        self.__original_functions = dict()
        self.__original_attributes = dict()

    def hijack(self, attribute):
        if attribute not in self.__original_functions:
            self.__original_functions[attribute] = getattr(self.__module, attribute)

        def decorator(function):
            def wrapper(*args, **kwargs):
                return function(*args, **kwargs, original_function=self.__original_functions[attribute])

            setattr(self.__module, attribute, wrapper)
            return function

        return decorator

    def hijack_attribute(self, attribute):
        def decorator(new_value):
            if attribute not in self.__original_attributes:
                self.__original_attributes[attribute] = getattr(self.__module, attribute), new_value

            setattr(self.__module, attribute, new_value)

        return decorator

    def get_original_attribute(self, attribute):
        return self.__original_attributes[attribute][0]

    def get_backup_attribute(self, attribute):
        return self.__original_attributes[attribute][1]

    def reset_module(self):
        for attribute, original_function in self.__original_functions.items():
            setattr(self.__module, attribute, original_function)
        self.__original_functions.clear()

        for attribute, (original_value, _) in self.__original_attributes.items():
            setattr(self.__module, attribute, original_value)
        self.__original_attributes.clear()

    @staticmethod
    def install_or_get(module, hijacker_attribute, on_uninstall=lambda _callback: None):
        if not hasattr(module, hijacker_attribute):
            module_hijacker = ModuleHijacker(module)
            setattr(module, hijacker_attribute, module_hijacker)
            on_uninstall(lambda: delattr(module, hijacker_attribute))
            on_uninstall(module_hijacker.reset_module)
            return module_hijacker
        else:
            return getattr(module, hijacker_attribute)
